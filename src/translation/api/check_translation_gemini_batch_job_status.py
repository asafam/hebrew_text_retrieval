import argparse
import time
from translation.api.translate_batch_gemini import check_batch_status


def main():
    parser = argparse.ArgumentParser(description="Poll status of Gemini batch translation jobs.")
    parser.add_argument("--sleep_time", type=int, default=60,
                        help="Seconds between status checks.")
    args = parser.parse_args()

    while True:
        jobs_metadata = check_batch_status()
        terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                           "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        if all(job["status"] in terminal_states for job in jobs_metadata):
            print("All jobs reached a terminal state.")
            break
        time.sleep(args.sleep_time)


if __name__ == "__main__":
    main()
