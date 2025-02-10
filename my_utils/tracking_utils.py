import os
def init_tracking_file(output_path, case_name):
    os.makedirs(output_path, exist_ok=True)
    tracking_file = output_path + '/'+ case_name + '.txt'
    with open(tracking_file, "w") as file:
        pass
    file.close()
    return tracking_file
