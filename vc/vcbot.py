import librosa
from fairseq import checkpoint_utils
import torch
import numpy as np
from scipy import signal
import parselmouth
import torch.nn.functional as F
import faiss

from .models import SynthesizerTrnMs256NSFsid
from .config import config

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
            torch.pow(rms1, torch.tensor(1 - rate))
            * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


def load_hubert(model_path: str):
    print("load hubert {}".format(model_path))
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()
    return hubert_model


def get_vc(model_path: str):
    print("loading %s" % model_path)
    cpt = torch.load(model_path, config.device)
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, cpt, net_g)
    return vc


class VC:
    def __init__(self, tgt_sr, cpt, net_g):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.tgt_sr = tgt_sr
        self.cpt = cpt
        self.net_g = net_g
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值

    def get_f0(
            self,
            x,
            p_len,
            f0_up_key
    ):
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0 = (
            parselmouth.Sound(x, self.sr)
            .to_pitch_ac(
                time_step=time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            )
            .selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(
                f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
            )
        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
                f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int_)
        return f0_coarse, f0bak  # 1-0

    def vc(
            self,
            model,
            sid,
            audio0,
            pitch,
            pitchf,
            index,
            big_npy,
            index_rate,
            # version,
            protect,
    ):  # ,file_index,file_big_npy
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(config.device).fill_(False)

        inputs = {
            "source": feats.to(config.device),
            "padding_mask": padding_mask,
            "output_layer": 9,
        }
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0])
        if protect < 0.5 and pitch != None and pitchf != None:
            feats0 = feats.clone()
        if (
                isinstance(index, type(None)) == False
                and isinstance(big_npy, type(None)) == False
                and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                    torch.from_numpy(npy).unsqueeze(0).to(config.device) * index_rate
                    + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch != None and pitchf != None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch != None and pitchf != None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        if protect < 0.5 and pitch != None and pitchf != None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=config.device).long()
        with torch.no_grad():
            if pitch != None and pitchf != None:
                audio1 = (
                    (self.net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (self.net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio1


class VCBot:
    def __init__(self, model_path: str, index_path: str, hubert_model: str):
        self.f0_up_key = 0  # 男转女推荐+12key, 女转男推荐-12key, 变调(整数, 半音数量, 升八度12降八度-12)
        self.index_rate = 0.88  # 检索特征占比
        self.protect = 0.33  # 保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果
        self.rms_mix_rate = 1  # 输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络
        self.resample_sr = 0  # 后处理重采样至最终采样率，0为不进行重采样
        self.sample_sr = 16000  # 采样率
        self.threshold = 0.02  # 音频是否无声阈值
        self.index_path = index_path
        self.hubert_model = load_hubert(hubert_model)
        self.vc = get_vc(model_path)
        self.isGPU = config.device == "cuda:0"

    def infer(self, audio: str | np.ndarray):
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=self.sample_sr)
        if librosa.feature.rms(y=audio).max() < self.threshold:
            print("voiceless !!!")
            return librosa.resample(audio, orig_sr=16000, target_sr=44100)
        else:
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            index = faiss.read_index(self.index_path)
            big_npy = index.reconstruct_n(0, index.ntotal)
            audio = signal.filtfilt(bh, ah, audio)
            audio_pad = np.pad(audio, (self.vc.window // 2, self.vc.window // 2), mode="reflect")
            opt_ts = []
            if audio_pad.shape[0] > self.vc.t_max:
                audio_sum = np.zeros_like(audio)
                for i in range(self.vc.window):
                    audio_sum += audio_pad[i: i - self.vc.window]
                for t in range(self.vc.t_center, audio.shape[0], self.vc.t_center):
                    opt_ts.append(
                        t
                        - self.vc.t_query
                        + np.where(
                            np.abs(audio_sum[t - self.vc.t_query: t + self.vc.t_query])
                            == np.abs(audio_sum[t - self.vc.t_query: t + self.vc.t_query]).min()
                        )[0][0]
                    )
            audio_opt = []
            t = None
            audio_pad = np.pad(audio, (self.vc.t_pad, self.vc.t_pad), mode="reflect")
            p_len = audio_pad.shape[0] // self.vc.window
            sid = torch.tensor(0, device=config.device).unsqueeze(0).long()
            pitch, pitchf = self.vc.get_f0(
                audio_pad,
                p_len,
                self.f0_up_key,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if config.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=config.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=config.device).unsqueeze(0).float()
            audio_opt.append(
                self.vc.vc(
                    self.hubert_model,
                    sid,
                    audio_pad[t:],
                    pitch,
                    pitchf,
                    index,
                    big_npy,
                    self.index_rate,
                    self.protect,
                )[self.vc.t_pad_tgt: -self.vc.t_pad_tgt]
            )
            audio_opt = np.concatenate(audio_opt)
            if self.rms_mix_rate != 1:
                audio_opt = change_rms(audio, self.sample_sr, audio_opt, self.vc.tgt_sr, self.rms_mix_rate)
            if 16000 <= self.resample_sr != self.vc.tgt_sr:
                self.sample_sr = self.resample_sr
                audio_opt = librosa.resample(
                    audio_opt, orig_sr=self.vc.tgt_sr, target_sr=self.resample_sr
                )
            del pitch, pitchf, sid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return audio_opt

