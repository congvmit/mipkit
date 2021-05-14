from termcolor import cprint


def print_warning(message):
    cprint(message, color="yellow")


def print_info(message):
    cprint(message, color="green")


def print_error(message):
    cprint(message, color="red")
