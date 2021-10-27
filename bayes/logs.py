def write_to_file(log, dataset):
    with open("../logs.txt", "a") as logs:
        logs.write(log)
        logs.write(" on dataset: " + dataset)
        logs.write("\n")
