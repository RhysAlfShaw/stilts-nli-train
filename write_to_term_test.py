import readline
import subprocess


def input_with_prefill(prompt, text):
    def hook():
        readline.insert_text(text)
        readline.redisplay()

    readline.set_pre_input_hook(hook)
    try:
        return input(prompt)
    finally:
        readline.set_pre_input_hook()


def main():
    while True:
        default_cmd = "echo Hello, world!"
        cmd = input_with_prefill("> ", default_cmd)
        if cmd.strip().lower() in ("exit", "quit"):
            break

        result = subprocess.run(
            cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")


if __name__ == "__main__":
    main()
