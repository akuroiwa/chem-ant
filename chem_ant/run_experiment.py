import argparse
import subprocess
import configparser
import shlex

def run_similarity_ant(config_file):
    option_types = {
        'int_options': ['loop', 'population', 'generation', 'experiment', 'generate', 'select'],
        'bool_options': ['include', 'deep-learning', 'rgression', 'json', 'verbose']
    }

    config = configparser.ConfigParser()
    config.read(config_file)

    cmd = generate_command(config, 'similarity-ant', option_types)
    subprocess.run(cmd, check=True)

def run_similarity_mcts(config_file):
    option_types = {
        'int_options': ['loop', 'experiment', 'iterationLimit', 'generate', 'select'],
        'bool_options': ['include', 'json', 'verbose']
    }

    config = configparser.ConfigParser()
    config.read(config_file)

    cmd = generate_command(config, 'similarity-mcts', option_types)
    subprocess.run(cmd, check=True)

def generate_command(config, section, option_types):
    cmd = [section]

    for opt_type, opt_names in option_types.items():
        for opt in opt_names:
            if opt in config[section]:
                value = config[section][opt]
                if opt_type == 'bool_options':
                    if config[section].getboolean(opt):
                        cmd.append(f'--{opt}')
                elif opt_type == 'int_options':
                    cmd.extend([f'--{opt}', str(config[section].getint(opt))])
                else:  # string_options
                    cmd.extend([f'--{opt}', shlex.quote(value)])

    return cmd


def main():
    """Main function to parse arguments and run experiments."""
    parser = argparse.ArgumentParser(description="Run similarity experiments")
    parser.add_argument("config_file", help="Configuration file for the experiment")
    parser.add_argument("-a", "--ant", action="store_true", help="Run similarity-ant experiment")
    parser.add_argument("-m", "--mcts", action="store_true", help="Run similarity-mcts experiment")
    args = parser.parse_args()

    if args.ant:
        run_similarity_ant(args.config_file)
    if args.mcts:
        run_similarity_mcts(args.config_file)

if __name__ == "__main__":
    main()
