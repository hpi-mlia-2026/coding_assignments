def evaluate_result(result, expected):
    if result != expected:
        print("\033[31mWARNING: Your function returned the wrong result!\033[0m")
    else: 
        print("\033[32mCorrect!\033[0m")