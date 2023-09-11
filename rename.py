import os

def rename_subfolders(target_name, new_name, start_path='.'):
    for root, dirs, files in os.walk(start_path):
        for name in dirs:
            if name == target_name:
                old_folder_path = os.path.join(root, name)
                new_folder_path = os.path.join(root, new_name)
                os.rename(old_folder_path, new_folder_path)
                print(f"Renamed folder {old_folder_path} to {new_folder_path}")

# Example usage: Uncomment the following line to run the function
rename_subfolders("100.0Pe_0.0Xi_8T5", "100.0Pe_0.0Xi_0T0", start_path='.')
