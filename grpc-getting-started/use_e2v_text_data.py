def doSomething():
    try:
        with open("sample1.json", "w", encoding="utf-8") as f:
            f.write("{\n")
            f.write("\t\"location\": {\n")
            f.write("\t\t\"latitude\": 407838,\n")
            f.write("\t\t\"longitude\": -74614\n")
            f.write("\t},\n")
            f.write("\t\"name\": \"Patriots Path, USA\"\n")
            f.write("}")
            return True
    except OSError as e:
        print(f"File write error: {e}")
        return False