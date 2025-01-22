import os
import re


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def read_files_from_directory(directory):
    files_content = {}
    for filename in sorted(os.listdir(directory), key=natural_sort_key):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                files_content[filename] = file.read()
    return files_content

def write_labeled_output(output_file, labels):
    with open(output_file, 'w', encoding='utf-8') as file:
        for label, file_numbers in labels.items():
            file.write(f"{label} {' '.join(file_numbers)}\n")

def main():
    input_directory = 'Result'
    output_file = 'label_output.txt'
    files_content = read_files_from_directory(input_directory)

    labels = {1: [], 2: [], 3: []}

    print("Press 1, 2, or 3 to label the files. Press 'q' to quit.")

    for filename, content in files_content.items():
        print(f"Content of {filename}:\n{content}\n")
        label = None
        while label not in ['1', '2', '3']:
            label = input("Enter label (1, 2, 3) or 'q' to quit: ")
            if label == 'q':
                print("Quitting...")
                write_labeled_output(output_file, labels)
                print("Labeled output has been written to", output_file)
                return
        labels[int(label)].append(str(int(filename.split('_')[1].split('.')[0])))

    write_labeled_output(output_file, labels)
    print("Labeled output has been written to", output_file)

if __name__ == "__main__":
    main()