import os
import subprocess

# MFA Configuration Constants
DEFAULT_EXPERIMENT_NAME = "PyPraat_sentences"
DEFAULT_DICTIONARY_PATH = "pretrained_models/dictionary/english_uk_mfa.dict"
DEFAULT_MODEL_PATH = "pretrained_models/acoustic/english_mfa.zip"
DEFAULT_CONFIG_PATH = ""


def run_mfa_alignment(
    *, input_dir="", output_dir="", dictionary_path="", model_path="", config_path=""
):
    """
    Run the Montreal Forced Aligner on the input directory containing .wav and .txt files.

    Parameters:
    - input_dir (str): Directory containing input .wav and corresponding .txt files.
    - output_dir (str): Directory to store the aligned output.
    - model (str): Acoustic model to use for alignment (default is 'english').

    Returns:
    - None
    """
    if not input_dir:
        input_dir = os.path.join(os.getcwd(), DEFAULT_EXPERIMENT_NAME)
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(
            f"Input directory '{input_dir}' not found. Please provide a valid input directory containing .wav and .txt files."
        )
        return
    if not output_dir:
        output_dir = os.path.join(input_dir, "aligned")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    if not dictionary_path:
        dictionary_path = os.path.join(os.getcwd(), DEFAULT_DICTIONARY_PATH)
    if not os.path.exists(dictionary_path):
        print(
            f"Dictionary file '{dictionary_path}' not found. Please provide a valid dictionary file."
        )
        return
    if not model_path:
        model_path = os.path.join(os.getcwd(), DEFAULT_MODEL_PATH)
    if not os.path.exists(model_path):
        print(
            f"Acoustic model '{model_path}' not found. Please provide a valid acoustic model file."
        )
        return
    if not config_path:
        if DEFAULT_CONFIG_PATH:
            config_path = os.path.join(os.getcwd(), DEFAULT_CONFIG_PATH)
    if config_path and not os.path.exists(config_path):
        print(f"Config file '{config_path}' not found. Using default config.")
    print(f"Input directory: {input_dir}")
    print(f"Dictionary path: {dictionary_path}")
    print(f"Acoustic model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Config path: {config_path}")

    # # Construct the MFA corpus validation command
    # validation_command = [
    #     "mfa", "validate",
    #     input_dir,        # Corpus Directory containing input files
    #     dictionary_path,  # Dictionary file
    #     # "--final_clean"   # Remove temporary files after alignment (removes wav and lab files too)
    # ]
    # # mfa validate [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH
    # try:
    #     # Run the MFA command
    #     subprocess.run(validation_command, check=True)
    #     print("Validation completed successfully. Corpus is ready for alignment.")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during validation: {e}")
    #     return
    # except FileNotFoundError:
    #     print("Montreal Forced Aligner (mfa) not found. Please ensure MFA is installed and accessible from the command line.")
    #     return

    # Construct the MFA alignment command
    if config_path:
        alignment_command = [
            "mfa",
            "align",
            input_dir,  # Corpus Directory containing input files
            dictionary_path,  # Dictionary file
            model_path,  # Acoustic model (e.g., "english")
            output_dir,  # Output directory
            "--config",
            "align_config.yaml",
            "--clean",  # Remove temporary files and force fresh alignment
        ]
    else:
        alignment_command = [
            "mfa",
            "align",
            input_dir,  # Corpus Directory containing input files
            dictionary_path,  # Dictionary file
            model_path,  # Acoustic model (e.g., "english")
            output_dir,  # Output directory
            "--clean",  # Remove temporary files and force fresh alignment
        ]
    print("Running alignment command:")
    print(" ".join(alignment_command))
    # mfa align [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY
    try:
        # Run the MFA command
        subprocess.run(alignment_command, check=True)
        print(
            f"Alignment completed successfully. Results are in the '{output_dir}' directory."
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during alignment: {e}")
    except FileNotFoundError:
        print(
            "Montreal Forced Aligner (mfa) not found. Please ensure MFA is installed and accessible from the command line."
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Montreal Forced Aligner alignment.")
    parser.add_argument("--input_dir", default="", help="Directory containing input .wav and .lab/.txt files.")
    parser.add_argument("--output_dir", default="", help="Directory to store aligned TextGrids.")
    parser.add_argument("--dictionary_path", default="", help="Path to pronunciation dictionary.")
    parser.add_argument("--model_path", default="", help="Path to acoustic model.")
    parser.add_argument("--config_path", default="", help="Path to alignment config YAML.")
    args = parser.parse_args()
    run_mfa_alignment(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dictionary_path=args.dictionary_path,
        model_path=args.model_path,
        config_path=args.config_path,
    )
