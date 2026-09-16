import subprocess


def volume_up():
    powershell_command = """
    $wsh = New-Object -ComObject WScript.Shell
    $wsh.SendKeys([char]175)
    """

    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            powershell_command
        ],
        capture_output=True,
        text=True
    )


print("Increasing Windows volume...")

for _ in range(5):
    volume_up()

print("Volume command sent successfully.")