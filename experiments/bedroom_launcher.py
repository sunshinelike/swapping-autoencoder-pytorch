from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher): #卧室加载器
    def options(self):
        opt = Options()
        opt.set(
            dataroot="~/datasets/lsun/bedroom_train_lmdb",
            dataset_mode="lmdb",
            num_gpus=8, batch_size=32,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="scale_shortside_and_crop",
            load_size=256, crop_size=256,
            display_freq=1600, print_freq=480,
        ),

        return [
            opt.specify(
                name="bedroom_default",
                patch_use_aggregation=False, #是否使用聚合
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=True,
            evaluation_metrics="swap_visualization", #评价指标
            evaluation_freq=50000, #评估频率
        ) for opt in common_options]

    def test_options_fid(self):
        return []

    def test_options(self):  #测试
        common_options = self.options()

        return [opt.tag("fig4").specify(  #tag?这里代表什么
            num_gpus=1,
            batch_size=1,
            dataroot="./testphotos/bedroom/fig4",
            dataset_mode="imagefolder",
            preprocess="scale_shortside",  # For testing, scale but don't crop 
            evaluation_metrics="structure_style_grid_generation",
        ) for opt in common_options]
