# Standard imports
import json
import os
import re

import pathspec

# Internal imports
from src.config.settings import PROJECT_ROOT


def load_gitignore(directory):
    """Loads and parses .gitignore file"""
    gitignore_path = os.path.join(directory, ".gitignore")
    if not os.path.exists(gitignore_path):
        return None

    with open(gitignore_path, encoding="utf-8") as f:
        return pathspec.PathSpec.from_lines("gitwildmatch", f)


def truncate_content(content, max_length=5000):
    """Truncates file content if it exceeds max_length"""
    if len(content) > max_length:
        return content[: max_length // 2] + "\n...\n" + content[-max_length // 2 :]
    return content


def strip_comments(content, file_extension):
    """Removes comments from Python and HTML files"""
    if file_extension == ".py":
        content = re.sub(r"(?m)^ *#.*\n?", "", content)
        content = re.sub(r"'''(.*?)'''", "", content, flags=re.DOTALL)
        content = re.sub(r'"""(.*?)"""', "", content, flags=re.DOTALL)
    elif file_extension == ".html":
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    return content


def read_file(file_path):
    """Reads file content with fallback encoding"""
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, encoding="latin-1") as f:
                return f.read()
        except UnicodeDecodeError:
            return None


def is_in_selected_apps(file_path, base_path, apps):
    """Checks if file belongs to the specified Django apps"""
    if apps is None:
        return True  # If no apps are specified, include all files

    for app in apps:
        app_path = os.path.join(base_path, app)
        if file_path.startswith(app_path):
            return True
    return False


def process_file(file_path, directory, gitignore):
    """Processes a file and returns its relative path and content"""
    relative_path = os.path.relpath(file_path, directory)

    if gitignore and gitignore.match_file(relative_path):
        return None

    if file_path.endswith((".py", ".html")):
        # file_extension = os.path.splitext(file_path)[1]
        content = read_file(file_path)

        if content:
            return {"path": relative_path, "content": content}

    return None


def get_file_structure(directory, gitignore, apps):
    """Recursively scans the project directory and processes files"""
    file_structure = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file belongs to the selected apps
            if is_in_selected_apps(file_path, directory, apps):
                file_info = process_file(file_path, directory, gitignore)
                if file_info:
                    file_structure.append(file_info)

    return file_structure


def save_to_json(data, file_path):
    """Saves serialized data to a JSON file"""
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, separators=(",", ":"))


def build_dependencies_files(list_files: list) -> dict:
    # base_path = os.getcwd()  # Set base_path to the current working directory
    gitignore = load_gitignore(PROJECT_ROOT)

    for ff in list_files:
        # relative_path = os.path.relpath(ff, PROJ_ROOT)
        relative_path = os.path.relpath(ff)
        if gitignore and gitignore.match_file(relative_path):
            return {}

        content = {}

        if ff.endswith((".py", ".html")):
            content = read_file(ff)

            if content:
                return {"path": ff.replace(str(PROJECT_ROOT), "."), "content": content}
        return content
