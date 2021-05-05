# %%
import mipkit
args_dict = mipkit.load_yaml_config('config2.yaml',
                                    to_args=True)
print(args_dict.model.backbone)
print(args_dict.model.num_classes)
# %%
