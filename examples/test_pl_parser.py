from mipkit.pl_utils import parse_args

args = parse_args()

from mipkit.utils import load_yaml_config, save_config_as_yaml

# save_config_as_yaml(args, '.', 'config_test')
conf = load_yaml_config("/Users/congvo/Workspace/mipkit/test/config_test_23-05-2021_00:19:31.yaml")
print(conf)
