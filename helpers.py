import os


def get_next_filename(base_filename, folder):
    """
    Генерує унікальну назву файлу, додаючи +1 до номера.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    files = os.listdir(folder)

    matching_files = [f for f in files if f.startswith(base_filename) and f.endswith(".csv")]

    max_number = 0
    for file in matching_files:
        try:
            number = int(file.replace(base_filename, "").replace(".csv", "").strip("_"))
            if number > max_number:
                max_number = number
        except ValueError:
            continue

    next_number = max_number + 1
    return os.path.join(folder, f"{base_filename}_{next_number}.csv")