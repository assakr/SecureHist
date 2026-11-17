import sys
import argparse

def main():
    print("=========================================================")
    print("     Unified Medical Image Watermarking Suite            ")
    print("=========================================================")
    
    parser = argparse.ArgumentParser(description="Run the Unified Watermarking Suite.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["dft", "dwt", "dwt-svd"], 
        help="Select the watermarking model to run."
    )
    
    args = parser.parse_args()
    
    choice = args.model
    if not choice:
        print("Please select a mathematical model for watermarking:")
        print("  1) DFT-Based (Discrete Fourier Transform)")
        print("  2) DWT-Based (Pure Discrete Wavelet Transform)")
        print("  3) DWT-SVD Hybrid (Wavelet + Singular Value Decomposition)")
        
        user_input = input("\nEnter 1, 2, or 3: ").strip()
        
        if user_input == "1":
            choice = "dft"
        elif user_input == "2":
            choice = "dwt"
        elif user_input == "3":
            choice = "dwt-svd"
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
            
    print(f"\n[+] Selected Model: {choice.upper()}\n")
    
    if choice == "dft":
        from model_dft import run_dft
        run_dft()
    elif choice == "dwt":
        from model_dwt import run_dwt_only
        run_dwt_only()
    elif choice == "dwt-svd":
        from model_dwt_svd import run_dwt_svd
        run_dwt_svd()

if __name__ == "__main__":
    main()
