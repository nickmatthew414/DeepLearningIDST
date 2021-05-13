import os


dirs = ["daytime_cities", "nighttime_cities"]

for dir in dirs:
    prefix = dir.split("_")[0]
    files = os.listdir(dir)
    # print(len(files))
    for i, file in enumerate(files):
        os.rename(os.path.join(dir, file), os.path.join(dir, "{}_{}.jpg".format(prefix, i+1)))