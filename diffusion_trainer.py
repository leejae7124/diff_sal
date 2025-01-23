import os
import time
import cv2, json
import numpy as np
import copy
import torch
from model import generate_av_model
from models.sal_losses import get_kl_cc_sim_loss_wo_weight, get_lossv2
from util.utils import (
    get_optim_scheduler,
    AverageMeter,
    Logger,
    normalize_data,
    LogWritter,
    AverageMeterList,
)
from models.diffusion_decoder.diffusion_utils import get_beta_schedule, to_torch
from datasets import data_transform, inverse_data_transform
from datasets.prepare_data import (
    get_training_loader,
    get_val_loader,
    get_val_av_loader,
    get_training_av_loader,
    get_test_av_loader,
)


class DiffusionTrainer(object):
    def __init__(self, args, config, device=None): #config: diffusion.yml
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.training_target = config.training.training_target #training_target: x0(원본 이미지)
        assert self.training_target in ["x0", "noise"]

        self.model, _ = generate_av_model(args)
        print(self.model)

        self.model_var_type = config.model.var_type #분산 타입 지정
        betas = get_beta_schedule( #베타 스케줄 설정
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = to_torch(betas).to(self.device)
        alphas = 1.0 - betas
        alphas_hat = alphas.cumprod(dim=0)
        alphas_hat_prev = torch.cat([torch.ones(1).to(device), alphas_hat[:-1]], dim=0)
        self.betas = betas
        self.alphas_hat = alphas_hat
        self.alphas_hat_prev = alphas_hat_prev
        self.sqrt_alphas_hat = torch.sqrt(alphas_hat)
        self.sqrt_one_minus_alphas_hat = torch.sqrt(1.0 - alphas_hat)
        self.log_one_minus_alphas_hat = torch.log(1.0 - alphas_hat)
        self.sqrt_recip_alphas_hat = torch.sqrt(1.0 / alphas_hat)
        self.sqrt_recipm1_alphas_hat = torch.sqrt(1.0 / alphas_hat - 1)

        posterior_variance = betas * (1.0 - alphas_hat_prev) / (1.0 - alphas_hat)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            torch.maximum(posterior_variance, torch.tensor(1e-20))
        ) #전체적인 diffusion model 파라미터를 설정하는 듯?
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_hat) / (1.0 - alphas_hat)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_hat_prev) * torch.sqrt(alphas) / (1.0 - alphas_hat)
        )

        self.num_timesteps = betas.shape[0]

    def prepare_data(self, data, targets, is_training=True):

        print(targets)
        if type(targets["salmap"]) == dict:
            targets["salmap"] = {
                key: targets["salmap"][key].cuda() for key in targets["salmap"]
            }
        else: #test일 때. 텐서를 gpu로 이동하고 float형으로 변환
            targets["salmap"] = targets["salmap"].cuda()
            targets["salmap"] = targets["salmap"].float()

        sal_maps = targets["salmap"]
        if len(sal_maps.shape) == 5:
            sal_maps = sal_maps.view(
                -1, sal_maps.shape[1], sal_maps.shape[3], sal_maps.shape[4]
            )

        while data["rgb"].size(0) < self.args.batch_size:
            tmp_size = self.args.batch_size - data["rgb"].size(0)
            img_list = []
            for _ in range(tmp_size):
                img_list.append(data["rgb"][-1, :])
            img_tensor = torch.stack(img_list, 0)
            data["rgb"] = torch.cat((data["rgb"], img_tensor), 0)

        imgs = (
            data["rgb"]
            .view(-1, data["rgb"].shape[1], data["rgb"].shape[3], data["rgb"].shape[4])
            .cuda()
        )
        batch_size = sal_maps.size(0)

        if is_training:
            x = data_transform(self.config, sal_maps)
            noise = torch.randn_like(x)
            t0 = np.random.randint(0, self.num_timesteps)
            t_tensor = torch.full(
                (batch_size,), t0, dtype=torch.int64, device=self.device
            )
            x_noisy = self.q_sample(x_start=x, t=t0, noise=noise)

            return imgs, x, x_noisy, t_tensor, noise
        else:
            #x.shape = (4, 3, 16, 224, 384) (배치 사이즈, 채널, frame 수, H, W)
            batch_size = data['rgb'].shape[0]  # 배치 크기
            height = data['rgb'].shape[3]     # 이미지 높이
            width = data['rgb'].shape[4]      # 이미지 너비

            # 채널 수를 1로 설정 (saliency map은 그레이스케일이라 채널이 1)
            channels = 1

            # 노이즈 생성
            noise = torch.randn(batch_size, channels, height, width).cuda()

            # print("noise shape:", noise.shape) #랜덤 노이즈 텐서 생성. sal_maps.shape를 사용하는 이유는 모델의 입력 크기에 맞추기 위해
            # noise = torch.randn(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]).cuda()
  # imgs 크기를 기준으로 노이즈 생성 #랜덤 노이즈 텐서 생성. sal_maps.shape를 사용하는 이유는 모델의 입력 크기에 맞추기 위해
            return imgs, noise

    def q_sample(self, x_start, t, noise=None):
        """
        Perform forward diffusion (noising) in a single step.
        This method returns x_t, which is x_0 noised for t timesteps.

        Args:
            x_start (torch.Tensor): Represents the original image (x_0).
            t (int): The timestep that measures the amount of noise to add.
        """
        
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            self.sqrt_alphas_hat[t] * x_start
            + self.sqrt_one_minus_alphas_hat[t] * noise
        )

    def train_av_data(self):
        """training audio-visual datasets"""
        args, config = self.args, self.config
        json_path = "cfgs/dataset.json"
        with open(json_path) as fp:
            data_config = json.load(fp)

        split_list = ["split1", "split2", "split3"]
        for split in split_list:
            result_path = "{}_results".format(split)
            weight_path = "{}_weights".format(split)
            args.result_path = os.path.join(args.root_path, result_path)
            args.weight_path = os.path.join(args.root_path, weight_path)

            os.makedirs(args.result_path, exist_ok=True)
            os.makedirs(args.weight_path, exist_ok=True)

            print("save path: {}".format(args.result_path))
            print("weight path: {}".format(args.weight_path))
            data_config["index"] = split

            num_epoches = self.config.training.n_epochs_for_av_data
            train_loader = get_training_av_loader(args, data_config)
            optimizer, scheduler = get_optim_scheduler(
                self.config, self.model.parameters(), num_epoches
            )

            train_model_log = LogWritter(
                os.path.join(args.result_path, "training_model.txt")
            )
            if self.args.rank == 0:
                train_model_log.update_txt(self.model.module, mode="w")

            train_logger = Logger(
                os.path.join(args.result_path, "train.log"),
                ["epoch", "total_step", "loss", "main", "cc", "sim", "nss", "lr"],
            )
            val_logger = Logger(
                os.path.join(args.result_path, "val.log"),
                ["epoch", "total_step", "loss", "main", "cc", "sim", "nss"],
            )

            start_epoch, step = 0, 0
            if args.resume_training:
                states = torch.load(
                    args.pretrain_path, map_location=torch.device("cpu")
                )
                msg = self.model.load_state_dict(states["state_dict"], strict=0)
                states["optim_dict"]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states["optim_dict"])
                start_epoch = states["epoch"]
                step = states["step"]

                del states
                print("resume training: {}/{}".format(args.pretrain_path, msg))

            self.config.training.snapshot_freq = len(train_loader)
            min_loss = 0.0
            for epoch in range(start_epoch, num_epoches):
                train_time = AverageMeter()
                data_time = AverageMeter()

                name_list = ["main", "cc", "sim", "nss", "total"]
                loss_metrics = AverageMeterList(name_list=name_list)

                data_start = time.time()
                for i, (data, targets) in enumerate(train_loader):
                    imgs, x, x_noisy, t_tensor, noise = self.prepare_data(data, targets)
                    data_time.update(time.time() - data_start)

                    self.model.train()
                    step += 1

                    data = {
                        "img": imgs,
                        "input": x_noisy,
                        "train_obj": x if self.training_target == "x0" else noise,
                        "audio": data["audio"].cuda(),
                    }
                    pred = self.model(data, t_tensor)
                    total_loss = get_lossv2(config, pred, data["train_obj"])

                    loss = total_loss["total"]
                    train_time.update(time.time() - data_start)

                    loss_metrics.update(total_loss)
                    optimizer.zero_grad()
                    loss.backward()

                    try:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass

                    optimizer.step()

                    data_start = time.time()
                    if (
                        self.args.rank == 0
                        and step % self.config.training.log_freq == 1
                    ):
                        self.print_train_info(
                            epoch,
                            num_epoches,
                            i,
                            train_loader,
                            step,
                            train_time,
                            data_time,
                            loss_metrics,
                            optimizer.param_groups[0]["lr"],
                            is_print=True,
                            logger=None,
                        )

                cur_val_loss = self.test_av_data_for_training(
                    data_config,
                    epoch=epoch + 1,
                    step=step,
                    val_logger=val_logger,
                    save_img=False,
                )
                states = {
                    "state_dict": self.model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "step": step,
                }
                torch.save(
                    states,
                    os.path.join(
                        self.args.weight_path, "ckpt_{}.pth".format(epoch + 1)
                    ),
                )

                if cur_val_loss > min_loss:
                    print(cur_val_loss, min_loss)
                    min_loss = cur_val_loss

                    torch.save(states, os.path.join(self.args.weight_path, "best.pth"))

                if self.args.rank == 0:
                    self.print_train_info(
                        epoch,
                        num_epoches,
                        i,
                        train_loader,
                        step,
                        train_time,
                        data_time,
                        loss_metrics,
                        optimizer.param_groups[0]["lr"],
                        is_print=False,
                        logger=train_logger,
                    )
                scheduler.step()

            print("Finish training!!!")

    def train(self):
        """traning dhf1k, holly, and ucf datasets"""
        args, config = self.args, self.config

        num_epoches = self.config.training.n_epochs
        train_loader = get_training_loader(args)
        optimizer, scheduler = get_optim_scheduler(
            self.config, self.model.parameters(), num_epoches
        )

        train_model_log = LogWritter(
            os.path.join(args.result_path, "training_model.txt")
        )

        if self.args.rank == 0:
            train_model_log.update_txt(self.model.module, mode="w")

        train_logger = Logger(
            os.path.join(args.result_path, "train.log"),
            ["epoch", "total_step", "loss", "main", "cc", "sim", "nss", "lr"],
        )
        val_logger = Logger(
            os.path.join(args.result_path, "val.log"),
            ["epoch", "total_step", "loss", "main", "cc", "sim", "nss"],
        )

        start_epoch, step = 0, 0
        if args.resume_training:
            states = torch.load(args.pretrain_path, map_location=torch.device("cpu"))
            msg = self.model.load_state_dict(states["state_dict"], strict=0)
            states["optim_dict"]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states["optim_dict"])
            start_epoch = states["epoch"]
            step = states["step"]

            del states
            print("resume training: {}/{}".format(args.pretrain_path, msg))

        self.config.training.snapshot_freq = len(train_loader)
        min_loss = 0.0
        for epoch in range(start_epoch, num_epoches):
            train_time = AverageMeter()
            data_time = AverageMeter()
            name_list = ["main", "cc", "sim", "nss", "total"]
            loss_metrics = AverageMeterList(name_list=name_list)

            data_start = time.time()
            for i, (data, targets) in enumerate(train_loader):
                imgs, x, x_noisy, t_tensor, noise = self.prepare_data(data, targets)
                data_time.update(time.time() - data_start)

                self.model.train()
                step += 1

                data = {
                    "img": imgs,
                    "input": x_noisy,
                    "train_obj": x if self.training_target == "x0" else noise,
                }
                pred = self.model(data, t_tensor)
                total_loss = get_lossv2(config, pred, data["train_obj"])
                loss = total_loss["total"]
                loss_metrics.update(total_loss)

                train_time.update(time.time() - data_start)

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass

                optimizer.step()

                data_start = time.time()
                if self.args.rank == 0 and step % self.config.training.log_freq == 1:
                    self.print_train_info(
                        epoch,
                        num_epoches,
                        i,
                        train_loader,
                        step,
                        train_time,
                        data_time,
                        loss_metrics,
                        optimizer.param_groups[0]["lr"],
                        is_print=True,
                        logger=None,
                    )
            if self.args.rank == 0:
                self.print_train_info(
                    epoch,
                    num_epoches,
                    i,
                    train_loader,
                    step,
                    train_time,
                    data_time,
                    loss_metrics,
                    optimizer.param_groups[0]["lr"],
                    is_print=False,
                    logger=train_logger,
                )

            states = {
                "state_dict": self.model.state_dict(),
                "optim_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "step": step,
            }
            torch.save(
                states,
                os.path.join(self.args.weight_path, "ckpt_{}.pth".format(epoch + 1)),
            )

            cur_val_loss = self.test(
                epoch=epoch + 1,
                step=step,
                val_logger=val_logger,
                save_img=False,
                is_testing=False,
            )
            if cur_val_loss > min_loss:
                min_loss = cur_val_loss
                torch.save(states, os.path.join(self.args.weight_path, "best.pth"))

            scheduler.step()

        print("Finish training!!!")

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            self.sqrt_recip_alphas_hat[t] * x_t - x0
        ) / self.sqrt_recipm1_alphas_hat[t]

    @torch.no_grad()
    def sample_ddim(self, x, img=None, audio_cond=None): #이걸 수행하게 됨.
        """ddim sampling"""
        #x = 초기 노이즈 텐서. 논문에서 St
        #img: visual conditions. 슈도코드에서 I.

        skip = self.num_timesteps // self.config.sampling.timesteps #건너뛸 간격. timesteps=1로 설정하여, sampling을 단 한 번! 수행하겠다는 의미. skip은 1000이 된다.
        seq = range(0, self.num_timesteps, skip) #seq: reverse process에서 거치는 timesteps sequence. 0부터 num_timesteps까지의 값.
        eta = self.config.sampling.eta #0으로 설정되어서 랜덤 노이즈 X

        tmp_img = copy.deepcopy(img)
        n = x.size(0) #배치 크기(noisy image의 첫 차원 크기)
        seq_next = [-1] + list(seq[:-1]) #다음 스텝

        #Reverse process 시작
        for time, time_next in zip(reversed(seq), reversed(seq_next)): #현재 time step과 다음 time step의 쌍을 가져온다.
            t_tensor = torch.full((n,), time, dtype=torch.int64, device=self.device) #현재 time step "t"를 모델에 전달할 텐서로 생성한다.
            img = copy.deepcopy(tmp_img)

            alpha = self.alphas_hat[time]
            alpha_next = self.alphas_hat[time_next]
            #x.shape = (4, 3, 16, 224, 384) (배치 사이즈, 채널, frame 수, H, W)
            if self.training_target == "x0": #이렇게 설정되어 있음. x0는 원본 이미지로, 원본 이미지를 예측하겠다.
                x_start = self.model.module.decoder_net(x, t_tensor, img, audio_cond)#Saliency-UNet이 x0를 예측한다.
                pred_noise = self.predict_noise_from_start(x, time, x_start)
            else:
                pred_noise = self.model.module.decoder_net(x, t_tensor, img, audio_cond)
                x_start = (
                    x - pred_noise * (1 - alpha).sqrt()
                ) / alpha.sqrt()  # Remove the noise at step t

            if time_next < 0:
                x = x_start
                continue

            c1 = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c2 = ((1 - alpha_next) - c1**2).sqrt()
            x = (
                self.sqrt_alphas_hat[time_next] * x_start
                + c1 * torch.randn_like(x)
                + c2 * pred_noise
            )

        return x

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas_hat[t] * x_t
            - self.sqrt_recipm1_alphas_hat[t] * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        )
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_mean_variance(self, x, t, img, clip_denoised=True):
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)

        if self.training_target == "x0":
            x_recon = self.model.module.decoder_net(x, t_tensor, img)
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.model.module.decoder_net(x, t_tensor, img)
            )

        if clip_denoised:
            x_recon.clamp(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, img, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(
            x, t, img, clip_denoised=clip_denoised
        )
        noise = (
            torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        )  # no noise when t == 0
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample_ddpm(self, x, img):
        """ddpm sampling"""
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        # 预先提取特征
        vis_list = self.model.module.visual_net(img)
        spatiotemp_feat_list = self.model.module.spatiotemp_net(vis_list[0])
        tmp_feat_list = copy.deepcopy(spatiotemp_feat_list)

        skip = self.num_timesteps // self.config.sampling.timesteps
        seq = range(0, self.num_timesteps, skip)
        for t in reversed(seq):
            x = self.p_sample(x, t, spatiotemp_feat_list)
            spatiotemp_feat_list = copy.deepcopy(tmp_feat_list)
        return x

    @torch.no_grad()
    def sample_image(self, x, img=None, audio=None, base_samples=None):
        classes = None
        #base_samples: 
        if base_samples is None: #base_samples: 샘플링에 사용되는 조건이나 초기 상태. 여기서는 명시X
            if classes is None:
                model_kwargs = {}
            else:
                model_kwargs = {"y": classes}
        else:
            model_kwargs = {"y": base_samples["y"], "low_res": base_samples["low_res"]}
            #model_kwargs: 모델의 추가 입력 또는 조건 데이터를 포함하는 딕셔너리 형태의 변수

        if self.model.module.audio_net:
            audio_feat, audio_feat_embed = self.model.module.forward_vggish(audio)
        else:
            audio_feat, audio_feat_embed = None, None

        if self.model.module.visual_net: #vis_list: visual features list
            vis_list = self.model.module.visual_net(img) #visualnet으로 img 처리
        else:
            vis_list = [
                torch.randn((audio_feat.shape[0], 768, 8, 7, 12), device=img.device),
                torch.randn((audio_feat.shape[0], 384, 8, 14, 24), device=img.device),
                torch.randn((audio_feat.shape[0], 192, 8, 28, 48), device=img.device),
                torch.randn((audio_feat.shape[0], 96, 8, 56, 96), device=img.device),
            ]
        tmp_feat_list = copy.deepcopy(vis_list)

        if self.config.sampling.sample_type == "ddim": #이걸로 되어있음. 샘플링(Revrese process) 타입: DDIM으로 빠른 샘플링 수행

            x = self.sample_ddim(x, tmp_feat_list, audio_feat_embed)
            

        elif self.config.sampling.sample_type == "ddpm": #DDPM은 t step에 대하여 모두 수행
            skip = self.num_timesteps // self.config.sampling.timesteps
            seq = range(0, self.num_timesteps, skip)
            for t in reversed(seq):
                x = self.p_sample(x, t, spatiotemp_feat_list)
                spatiotemp_feat_list = copy.deepcopy(tmp_feat_list)

        elif self.config.sampling.sample_type in ["dpmsolver", "dpmsolver++"]:
            from models.dpm_solver.sampler import (
                NoiseScheduleVP,
                model_wrapper,
                DPM_Solver,
            )

            assert (
                self.training_target == "x0"
            ), "Need to use noise training objective!!!"

            def model_fn(x, t, vis_feat, **model_kwargs):
                out = self.model.module.decoder_net(x, t, vis_feat, **model_kwargs)
                return out

            noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
            model_fn_continuous = model_wrapper(
                model_fn,
                noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=0.0,
                classifier_fn=None,
                classifier_kwargs={},
            )
            dpm_solver = DPM_Solver(
                model_fn_continuous,
                noise_schedule,
                algorithm_type=self.config.sampling.sample_type,
                correcting_x0_fn=(
                    "dynamic_thresholding"
                    if self.config.sampling.thresholding
                    else None
                ),
            )
            x = dpm_solver.sample(
                x,
                spatiotemp_feat_list,
                steps=(
                    self.config.sampling.timesteps - 1
                    if self.config.sampling.denoise
                    else self.config.sampling.timesteps
                ),
                order=self.config.sampling.dpm_solver_order,
                skip_type=self.config.sampling.skip_type,
                method=self.config.sampling.dpm_solver_method,
                lower_order_final=self.config.sampling.lower_order_final,
                denoise_to_zero=self.config.sampling.denoise,
                solver_type=self.config.sampling.dpm_solver_type,
                atol=self.config.sampling.dpm_solver_atol,
                rtol=self.config.sampling.dpm_solver_rtol,
            )
            return x
        else:
            raise NotImplementedError

        return x

    @torch.no_grad()
    def val(self, epoch=0, step=0, val_logger=None, save_img=True):
        """validation"""
        print("Starting goning into validation phase!!!")
        args, config = self.args, self.config
        losses = AverageMeter()
        cc = AverageMeter()
        kl = AverageMeter()
        sim = AverageMeter()
        nss = AverageMeter()

        val_loader = get_val_loader(args)

        with torch.no_grad():
            for i, (data, targets) in enumerate(val_loader):
                imgs, x = self.prepare_data(data, targets)
                self.model.eval()

                batch_size = x.size(0)
                x = data_transform(self.config, x)
                noise = torch.randn_like(x)
                t0 = np.random.randint(0, self.num_timesteps)
                t_tensor = torch.full(
                    (batch_size,), t0, dtype=torch.int64, device=self.device
                )
                x_noisy = self.q_sample(x_start=x, t=t0, noise=noise)

                data = {"img": imgs, "input": x_noisy}
                if self.training_target == "x0":
                    x0_recon = self.model(data, t_tensor)
                    total_loss = get_lossv2(config, x0_recon, x)
                else:
                    noise_recon = self.model(data, t_tensor)
                    total_loss = get_lossv2(config, noise_recon, noise)

                kl.update(round(total_loss["main"].item(), 3))
                cc.update(round(total_loss["cc"].item(), 3))
                sim.update(round(total_loss["sim"].item(), 3))
                nss.update(round(total_loss["nss"].item(), 3))
                loss = total_loss["total"]
                losses.update(round(loss.item(), 3))

        if val_logger != None and args.rank == 0:
            print(
                "Val Epoch: [{0}][{1}/{2}]\t"
                "Loss {losses.val:.3f} ({losses.avg:.3f})\t"
                "CC {cc.val:.3f} ({cc.avg:.3f})\t"
                "SIM {sim.val:.3f} ({sim.avg:.3f})\t"
                "NSS {nss.val:.3f} ({nss.avg:.3f})\t"
                "MAIN {kl.val:.3f} ({kl.avg:.3f})".format(
                    epoch,
                    i + 1,
                    len(val_loader),
                    losses=losses,
                    cc=cc,
                    sim=sim,
                    nss=nss,
                    kl=kl,
                )
            )
            val_logger.log(
                {
                    "epoch": epoch,
                    "total_step": step,
                    "loss": round(losses.avg, 3),
                    "main": round(kl.avg, 3),
                    "cc": round(cc.avg, 3),
                    "sim": round(sim.avg, 3),
                    "nss": round(nss.avg, 3),
                }
            )

    @torch.no_grad()
    def test(self, epoch=0, step=0, val_logger=None, save_img=True, is_testing=True):
        """test visual dataset"""
        print("Start testing!!!")
        #root_path: experiments_on_dhf1k/visual
        args, config = self.args, self.config #config는 디퓨전 설정

        result_path = f"{args.data_type}_test_samplings/multi" #경로 생성? dhf1k(data_type)_test_samplings/multi
        result_path = os.path.join(args.root_path, result_path)
        weight_path = os.path.join(args.root_path, "weights", "best.pth") #가중치 경로 생성
        os.makedirs(result_path, exist_ok=True)

        args.pretrain_path = weight_path
        if is_testing and args.pretrain_path.strip() != "":
            states = torch.load(args.pretrain_path, map_location=torch.device("cpu"))
            msg = self.model.load_state_dict(states["state_dict"], strict=0) #모델 가중치 로드
            print("testing: {}/{}".format(args.pretrain_path, msg))
        name_list = ["main", "cc", "sim", "nss", "total"]
        loss_metrics = AverageMeterList(name_list=name_list) #계산을 위해 필요한 것 같음
        val_loader = get_val_loader(args)

        for i, (data, targets) in enumerate(val_loader):
            imgs, x_noise = self.prepare_data(data, targets, is_training=False)
            self.model.eval()

            pred = self.sample_image(x_noise, img=imgs) #여기까지 옴
            pred = inverse_data_transform(config, pred)

            # total_loss = get_kl_cc_sim_loss_wo_weight(
            #     config, pred, targets["salmap"].to(pred.device)
            # )
            # loss_metrics.update(total_loss)

            # if self.args.rank == 0 and i % self.config.training.log_freq == 1:
            #     self.print_val_info(
            #         epoch, i, val_loader, step, loss_metrics, is_print=True, logger=None
            #     )

            if save_img:
                self.save_img(data, pred, result_path)

        if val_logger != None and args.rank == 0:
            self.print_val_info(
                epoch,
                i,
                val_loader,
                step,
                loss_metrics,
                is_print=False,
                logger=val_logger,
            )

        return loss_metrics.get_metric("total").avg

    @torch.no_grad()
    def test_av_data_for_training(
        self,
        data_config,
        epoch=0,
        step=0,
        val_logger=None,
        save_img=True,
        is_testing=True,
    ):

        print("Start testing!!!")
        args, config = self.args, self.config

        split = data_config["index"]
        result_path = f"{split}_results"
        result_path = os.path.join(args.root_path, result_path)
        os.makedirs(result_path, exist_ok=True)

        name_list = ["main", "cc", "sim", "nss", "total"]
        loss_metrics = AverageMeterList(name_list=name_list)

        val_loader = get_val_av_loader(args, data_config)
        for i, (data, targets) in enumerate(val_loader):
            imgs, x_noise = self.prepare_data(data, targets, is_training=False)
            self.model.eval()
            audio = data["audio"].cuda()
            pred = self.sample_image(x_noise, img=imgs, audio=audio)
            pred = inverse_data_transform(config, pred)

            total_loss = get_kl_cc_sim_loss_wo_weight(
                config, pred, targets["salmap"].to(pred.device)
            )

            loss_metrics.update(total_loss)

            if self.args.rank == 0 and i % self.config.training.log_freq == 1:
                self.print_val_info(
                    epoch, i, val_loader, step, loss_metrics, is_print=True, logger=None
                )

            if save_img:
                self.save_img(data, pred, result_path)

        if val_logger != None and args.rank == 0:
            self.print_val_info(
                epoch,
                i,
                val_loader,
                step,
                loss_metrics,
                is_print=False,
                logger=val_logger,
            )
        return loss_metrics.get_metric("total").avg

    @torch.no_grad()
    def test_av_data(self, epoch=0, save_img=True):
        """test audio-visual dataset"""
        print("Start testing!!!")
        args, config = self.args, self.config

        json_path = "cfgs/dataset.json"
        with open(json_path) as fp:
            data_config = json.load(fp)

        split_list = ["split1", "split2", "split3"]
        for split in split_list:
            data_config["index"] = split
            weight_path = "{}_weights".format(split)
            weight_path = os.path.join(args.root_path, weight_path)

            result_path = f"{split}_results"
            result_path = os.path.join(args.root_path, result_path)
            os.makedirs(result_path, exist_ok=True)

            test_logger = Logger(
                result_path + ".log",
                ["epoch", "total_step", "loss", "main", "cc", "sim", "nss"],
            )

            args.pretrain_path = os.path.join(weight_path, "best.pth")
            if args.pretrain_path.strip() != "":
                states = torch.load(
                    args.pretrain_path, map_location=torch.device("cpu")
                )
                msg = self.model.load_state_dict(states["state_dict"], strict=0)
                print("testing: {}/{}".format(args.pretrain_path, msg))

            name_list = ["main", "cc", "sim", "nss", "total"]
            loss_metrics = AverageMeterList(name_list=name_list)
            test_loader = get_test_av_loader(args, data_config)

            for i, (data, targets) in enumerate(test_loader):
                imgs, x_noise = self.prepare_data(data, targets, is_training=False)
                self.model.eval()

                audio = data["audio"].cuda()
                pred = self.sample_image(x_noise, img=imgs, audio=audio)
                pred = inverse_data_transform(config, pred)

                total_loss = get_kl_cc_sim_loss_wo_weight(
                    config, pred, targets["salmap"].to(pred.device)
                )
                loss_metrics.update(total_loss)

                if self.args.rank == 0 and i % self.config.training.log_freq == 1:
                    self.print_val_info(
                        epoch,
                        i,
                        test_loader,
                        0,
                        loss_metrics,
                        is_print=True,
                        logger=None,
                    )

                if save_img:
                    self.save_img(data, pred, result_path, av_data=True)

            if test_logger != None and args.rank == 0:
                self.print_val_info(
                    epoch,
                    i,
                    test_loader,
                    0,
                    loss_metrics,
                    is_print=False,
                    logger=test_logger,
                )

    def save_img(self, data, pred, save_root, av_data=False):
        """Save the predicted data according to the data reading mode of dhf1k

        Args:
            data (_type_): _description_
            pred (_type_): _description_
            save_path (_type_): _description_
        """
        # print("data!!!", data.items())
        video_ids = data["video_index"]
        frame_ids = data["video_id"]
        print("id!! ", data["video_id"])
        # print("video_ids: ", video_ids)
  
        # gt_indexes = data["gt_index"].cpu().numpy()
        pred = pred.detach().cpu().numpy()
        
        # gt_indexes = gt_indexes.reshape(pred.shape[0], 1)

        if av_data:
            data_convert_dict = {
                "AVAD": "avad",
                "Coutrot_db1": "coutrot1",
                "Coutrot_db2": "coutrot2",
                "DIEM": "diem",
                "ETMD_av": "etmd",
                "SumMe": "summe",
            }

        # assert pred.shape[0] == len(video_ids) == len(gt_indexes)

        assert pred.shape[0] == len(video_ids) == len(frame_ids)
        for i, (img, vid, fid) in enumerate(zip(pred, video_ids, frame_ids)):
            if av_data:
                vid = str(
                    data_convert_dict[vid.split("/")[0]] + "/" + vid.split("/")[-1]
                )
                img_name = "pred_sal_{0:06d}.jpg".format(int(gid))
            else:
                vid = str(vid)
                # img_name = str(int(gid)) + ".png"
                img_name = str(fid)

            save_path = os.path.join(save_root, vid)
            print("save path: ", save_path)
            os.makedirs(save_path, exist_ok=True)
            sal_img = normalize_data(img.transpose(1, 2, 0))
            cv2.imwrite("{}/{}".format(save_path, img_name), sal_img)

    def print_train_info(
        self,
        epoch,
        num_epoches,
        i,
        train_loader,
        step,
        train_time,
        data_time,
        loss_metrics,
        lr,
        is_print=True,
        logger=None,
    ):
        if is_print:
            print(
                "Epoch: [{0}/{1}][{2}/{3}]\t"
                "Total Step: {4}\t"
                "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data: {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss: {loss.val:.4f} ({loss.avg:.4f})\t"
                "CC {cc.val:.3f} ({cc.avg:.3f})\t"
                "SIM {sim.val:.3f} ({sim.avg:.3f})\t"
                "NSS {nss.val:.3f} ({nss.avg:.3f})\t"
                "MAIN {kl.val:.3f} ({kl.avg:.3f})".format(
                    int(epoch) + 1,
                    int(num_epoches),
                    int(i) + 1,
                    len(train_loader),
                    int(step),
                    batch_time=train_time,
                    data_time=data_time,
                    loss=loss_metrics.get_metric("total"),
                    cc=loss_metrics.get_metric("cc"),
                    sim=loss_metrics.get_metric("sim"),
                    nss=loss_metrics.get_metric("nss"),
                    kl=loss_metrics.get_metric("main"),
                )
            )
        else:
            logger.log(
                {
                    "epoch": epoch + 1,
                    "total_step": step,
                    "loss": round(loss_metrics.get_metric("total").avg, 3),
                    "main": round(loss_metrics.get_metric("main").avg, 3),
                    "cc": round(loss_metrics.get_metric("cc").avg, 3),
                    "sim": round(loss_metrics.get_metric("sim").avg, 3),
                    "nss": round(loss_metrics.get_metric("nss").avg, 3),
                    "lr": lr,
                }
            )

    def print_val_info(
        self, epoch, i, val_loader, step, loss_metrics, is_print=True, logger=None
    ):
        if is_print:
            print(
                "Test Epoch: [{0}][{1}/{2}]\t"
                "CC {cc.val:.3f} ({cc.avg:.3f})\t"
                "SIM {sim.val:.3f} ({sim.avg:.3f})\t"
                "NSS {nss.val:.3f} ({nss.avg:.3f})\t"
                "MAIN {kl.val:.3f} ({kl.avg:.3f})".format(
                    epoch,
                    i + 1,
                    len(val_loader),
                    cc=loss_metrics.get_metric("cc"),
                    sim=loss_metrics.get_metric("sim"),
                    nss=loss_metrics.get_metric("nss"),
                    kl=loss_metrics.get_metric("main"),
                )
            )
        else:
            logger.log(
                {
                    "epoch": epoch,
                    "total_step": step,
                    "loss": round(loss_metrics.get_metric("total").avg, 3),
                    "main": round(loss_metrics.get_metric("main").avg, 3),
                    "cc": round(loss_metrics.get_metric("cc").avg, 3),
                    "sim": round(loss_metrics.get_metric("sim").avg, 3),
                    "nss": round(loss_metrics.get_metric("nss").avg, 3),
                }
            )
