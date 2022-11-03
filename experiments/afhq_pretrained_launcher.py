from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="/mnt/localssd/datasets/afhq/afhq/train",#目录
            dataset_mode="imagefolder",#数据模板
            checkpoints_dir="./checkpoints/",#预训练
            num_gpus=8, batch_size=32, #这里要更改，根据情况改
            # scale the image such that the short side is |load_size|, and #取短边
            # crop a square window of |crop_size|. #裁剪为正方形
            preprocess="resize", #预处理操作，归一化图像尺寸
            load_size=256, crop_size=256,#所以图像尺寸为256*256
        )

        return [
            opt.specify(
                name="afhq_pretrained",
            ),
        ]

    def train_options(self): 
        common_options = self.options()
        return [opt.specify(
            display_freq=1600, print_freq=480,#展示频率设置，打印频率设置
            continue_train=True, #持续训练
            evaluation_metrics="none") for opt in common_options] #评估指标
        
    def test_options(self):
        opt = self.options()[0]
        return [
            # Swapping Grid Visualization. Fig 12 of the arxiv paper
            opt.tag("swapping_grid").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="./testphotos/afhq/", #测试集存放
                dataset_mode="imagefolder",   #数据集模式
                evaluation_metrics="structure_style_grid_generation" #评价指标
            ),
            
            # Simple Swapping code for quick testing
            opt.tag("simple_swapping").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                evaluation_metrics="simple_swapping",
                # Specify the two images here. #指定测试图片
                input_structure_image="./testphotos/afhq/structure/flickr_dog_000846.jpg", #结构图
                input_texture_image="./testphotos/afhq/style/flickr_wild_001319.jpg", #纹理图
                # alpha == 1.0 corresponds to full swapping. #1.0代表全交换
                # 0 < alpha < 1 means interpolation #这期间的表示插值
                texture_mix_alpha=1.0,
            ),
            
            # Simple interpolation images for quick testing
            opt.tag("simple_interpolation").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.#这里不需要数据根目录是因为不需要训练，只测试
                evaluation_metrics="simple_swapping",
                # Specify the two images here.
                input_structure_image="./testphotos/afhq/structure/flickr_dog_000846.jpg",
                input_texture_image="./testphotos/afhq/style/flickr_wild_001319.jpg",
                texture_mix_alpha='0.0 0.25 0.5 0.75 1.0',
            )
        ]
