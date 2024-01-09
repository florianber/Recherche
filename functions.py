from colorama import Fore, Style
import inspect
import os

def print_colored(variable, color):
    color_map = {
        "blue": Fore.BLUE,
        "red": Fore.RED,
        "green": Fore.GREEN
    }

    if color not in color_map:
        print("Couleur non supportée.")
        return

    color_code = color_map[color]
    reset_code = Style.RESET_ALL

    frame = inspect.currentframe().f_back
    variable_name = [name for name, value in frame.f_locals.items() if value is variable][0]

    print(f"{color_code}{variable_name} = {variable}{reset_code}")

def empty_folder(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def exist_model(folder_path):
    files = os.listdir(folder_path)
    if 'model.pth' in files:
        return True
    else : 
        return False
    