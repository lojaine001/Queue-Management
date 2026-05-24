import re
with open("LOGs/systems.log") as f:
    for line in f:
        if "19:1" in line and ("COUNTER" in line or "DB" in line):
            print(line, end="")
