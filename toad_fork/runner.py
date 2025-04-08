import signal
import subprocess
import sys
import threading
import time
from queue import Queue
import torch

assert torch.cuda.is_available()
print("CUDA is available")


# Configurable parameters
COMMAND = "LOCAL_RUNNER=1 $HOME/.local/bin/luxai-s3 ./final_versions/08_03_tune_against_mask_cont_TEST/main.py ./final_versions/08_03_tune_against_mask_cont/main.py"  # Command to execute
NUM_THREADS = 2  # Number of threads

# Thread-safe queue to store results
result_queue = Queue()
completed_count = 0
lock = threading.Lock()

# Event to signal threads to stop
stop_event = threading.Event()

seeda = 10000000

def run_command_infinite(cmd):
    global completed_count
    global seeda
    while not stop_event.is_set() and completed_count < 1000:
        with lock:
            seeda += 1
            cmd_with_seed = f"{cmd} --seed {seeda}"
        try:
            # Execute the command
            process = subprocess.Popen(
                cmd_with_seed, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = process.communicate()

            # Put the result in the queue
            result_queue.put((cmd_with_seed, stdout, stderr, process.returncode))

            # Update the counter thread-safely
            with lock:
                completed_count += 1
                print(f"Completed: {completed_count} commands")
        except Exception as e:
            # Handle any errors
            result_queue.put((cmd_with_seed, "", str(e), -1))

def main():
    # Create and start threads
    threads = []
    for _ in range(NUM_THREADS):
        thread = threading.Thread(target=run_command_infinite, args=(COMMAND,))
        thread.daemon = True  # Ensures threads are killed when the main process exits
        thread.start()
        threads.append(thread)

    def signal_handler(sig, frame):
        print("\nExiting...")
        stop_event.set()  # Signal threads to stop
        for thread in threads:
            thread.join()  # Wait for all threads to finish
        sys.exit(0)

    # Register the signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)

    wins_0 = 0
    wins_1 = 0
    p_wins_0 = 0
    p_wins_1 = 0

    try:
        # Continuously process results
        while not stop_event.is_set():
            while not result_queue.empty():
                cmd, stdout, stderr, returncode = result_queue.get()
                #print(f"\nCommand: {cmd}")
                #print(f"Return Code: {returncode}")
                #print(f"Stdout:\n{stdout}")
                #print(f"Stderr:\n{stderr}")
                print('stderr: ', len(stderr), 'stdout:', len(stdout))
                has_res = False
                for line in stdout.split("\n"):
                    if "Rewards" in line:
                        points_0 = int(line.split('array(')[1].split(',')[0])
                        points_1 = int(line.split('array(')[2].split(',')[0])
                        assert points_0 + points_1 == 5
                        if points_0 > points_1:
                            wins_0 += 1
                        else:
                            wins_1 += 1

                        p_wins_0 += points_0
                        p_wins_1 += points_1

                        has_res = True
                        print(cmd)
                        print("Result:", points_0, points_1, "      Wins:", wins_0, wins_1, wins_0 + wins_1,  f"      Winrate_0: {(wins_0 / (wins_0 + wins_1) * 100):.2f}%", f"     Winrate_1: {(wins_1 / (wins_0 + wins_1) * 100):.2f}%")
                        print("Points:", ' ', ' ', "    P_Wins:", p_wins_0, p_wins_1, p_wins_0 + p_wins_1,  f"     P_Winrate_0: {(p_wins_0 / (p_wins_0 + p_wins_1) * 100):.2f}%", f"   P_Winrate_1: {(p_wins_1 / (p_wins_0 + p_wins_1) * 100):.2f}%")

                    print("?", line)
                if not has_res:
                    print(cmd)
                    print("No result")
                    print(f"Stderr:\n{stderr}")
                    print("")
                    stop_event.set()

            time.sleep(0.1)  # Small delay to avoid busy-waiting
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
