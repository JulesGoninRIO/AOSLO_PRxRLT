import traceback
import logging
import os
import sys

# sys.path.append(os.path.join("..", ".."))
from src.InputData_Pipe.PipelineEngine.PipelineRunner import PipelineRunner


def main():
    """
    Function that handles the run of the automation pipeline, set up the Log files
    and catch higher-level exceptions that are thrwon during the process
    """

    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("\nSTARTING AOImgProc PIPELINE\n")
    print("AOImgProc Pipeline has started. LISTENING...")

    # Choose whether you will run the pipeline on the Virtual Machine accessed
    # via the remote desktop or on local computer (Configs\config_laptop.txt)
    configs_file = r"config_VM.txt"

    pipeline = PipelineRunner(configs_file)

    try:
        pipeline.go()
    except Exception as e:
        # No unhandled exceptions should arrive here
        traceback.print_exc()
        logging.critical("CRITICAL ERROR OCCURRED!")
        logging.exception("Exception occurred",
                          exc_info=True)  # log stack trace
        pass  # TODO set up handler to send email alert
        # pipeline.flushPipeline() #do not flush for now, let's investigate errors
        logging.critical("STOPPING PIPELINE")

        # SWITCH BETWEEN XT (= 300 frames) AND PHOTORECEPTOR or other (= 100 frames)

        # XT coordinates workfow:
        # - uncomment, set oldest first

        # Power Automate Desktop workflow:
        # -XT: Button "Auto select reference", then "Wait 20sec" ; Button "Register all", then "Wait 30sec"
        # -Photoreceptor: Button "Auto select reference", then "Wait 10sec" ; Button "Register all", then "Wait 15sec"


if __name__ == "__main__":
    if not os.path.exists("./Logs"):
        os.mkdir("./Logs")
    logging.basicConfig(filename='Logs/app.log',
                        level=logging.DEBUG,
                        filemode='a',
                        format='%(asctime)s -%(levelname)s - %(message)s')
    main()
