from os import path
import os


def make_dir(pathname):
    if not path.isdir(pathname):
        os.makedirs(pathname, exist_ok=True)
