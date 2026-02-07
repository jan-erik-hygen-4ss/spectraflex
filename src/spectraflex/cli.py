"""Command-line interface for spectraflex.

Provides CLI commands for common operations:
- identify: Identify transfer functions from simulation files
- predict: Predict response statistics for a given sea state
- generate: Generate white noise model files
- library: Manage transfer function libraries
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="spectraflex",
        description="Transfer function identification and spectral response prediction",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # identify command
    identify_parser = subparsers.add_parser(
        "identify",
        help="Identify transfer functions from simulation or time histories",
    )
    identify_parser.add_argument(
        "input",
        help="Input file (.sim, .npz spectra, or .npz time histories)",
    )
    identify_parser.add_argument(
        "-o",
        "--output",
        help="Output file (.nc for NetCDF)",
        default=None,
    )
    identify_parser.add_argument(
        "--nperseg",
        type=int,
        default=1024,
        help="FFT segment length (default: 1024)",
    )
    identify_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help='Configuration as JSON string, e.g., \'{"hs": 2.0, "heading": 0}\'',
    )

    # predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict response statistics from transfer function and wave spectrum",
    )
    predict_parser.add_argument(
        "transfer_function",
        help="Transfer function file (.nc)",
    )
    predict_parser.add_argument(
        "--hs",
        type=float,
        required=True,
        help="Significant wave height [m]",
    )
    predict_parser.add_argument(
        "--tp",
        type=float,
        required=True,
        help="Peak period [s]",
    )
    predict_parser.add_argument(
        "--gamma",
        type=float,
        default=3.3,
        help="JONSWAP gamma (default: 3.3)",
    )
    predict_parser.add_argument(
        "--duration",
        type=float,
        default=10800.0,
        help="Duration for MPM calculation [s] (default: 10800 = 3 hours)",
    )
    predict_parser.add_argument(
        "-o",
        "--output",
        help="Output file for results (.json)",
        default=None,
    )

    # generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate white noise model files",
    )
    generate_parser.add_argument(
        "template",
        help="Template OrcaFlex model (.dat)",
    )
    generate_parser.add_argument(
        "--hs",
        type=float,
        nargs="+",
        default=[2.0],
        help="Significant wave height(s) [m]",
    )
    generate_parser.add_argument(
        "--direction",
        type=float,
        nargs="+",
        default=[0.0],
        help="Wave direction(s) [deg]",
    )
    generate_parser.add_argument(
        "--freq-range",
        type=float,
        nargs=2,
        default=[0.02, 0.25],
        metavar=("MIN", "MAX"),
        help="Frequency range [Hz] (default: 0.02 0.25)",
    )
    generate_parser.add_argument(
        "--duration",
        type=float,
        default=512.0,
        help="Simulation duration [s] (default: 512)",
    )
    generate_parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Output directory (default: current)",
    )

    # library command
    library_parser = subparsers.add_parser(
        "library",
        help="Manage transfer function libraries",
    )
    library_subparsers = library_parser.add_subparsers(dest="library_command")

    # library info
    library_info = library_subparsers.add_parser(
        "info",
        help="Show library information",
    )
    library_info.add_argument("library_file", help="Library file (.nc)")

    # library build
    library_build = library_subparsers.add_parser(
        "build",
        help="Build library from spectra files",
    )
    library_build.add_argument(
        "spectra_dir",
        help="Directory containing *_spectra.npz files",
    )
    library_build.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output library file (.nc)",
    )

    args = parser.parse_args(argv)

    if args.version:
        from spectraflex import __version__

        print(f"spectraflex {__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "identify":
        return cmd_identify(args)
    elif args.command == "predict":
        return cmd_predict(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "library":
        return cmd_library(args)

    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    """Handle identify command."""
    from spectraflex import identify
    from spectraflex.io import save_transfer_function

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    config = None
    if args.config:
        try:
            config = json.loads(args.config)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON config: {e}", file=sys.stderr)
            return 1

    print(f"Identifying transfer function from: {input_path}")

    try:
        if input_path.suffix == ".sim":
            # Need result specifications - for now, error out
            print(
                "Error: .sim files require result specifications. "
                "Use Python API or provide spectra file.",
                file=sys.stderr,
            )
            return 1
        elif input_path.suffix == ".npz":
            # Check if it's a spectra file or time history file
            tf = identify.from_spectra(input_path, config=config)
        else:
            print(f"Error: Unknown file type: {input_path.suffix}", file=sys.stderr)
            return 1

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_suffix(".nc")

        save_transfer_function(tf, output_path)
        print(f"Saved transfer function to: {output_path}")

        # Print summary
        print(f"  Frequencies: {len(tf.coords['frequency'])} points")
        print(f"  Variables: {list(tf.coords['variable'].values)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Handle predict command."""
    from spectraflex import predict, spectrum
    from spectraflex.io import load_transfer_function

    tf_path = Path(args.transfer_function)

    if not tf_path.exists():
        print(f"Error: Transfer function file not found: {tf_path}", file=sys.stderr)
        return 1

    print(f"Loading transfer function from: {tf_path}")

    try:
        tf = load_transfer_function(tf_path)

        # Create wave spectrum
        f = tf.coords["frequency"].values
        wave = spectrum.jonswap(hs=args.hs, tp=args.tp, f=f, gamma=args.gamma)

        print(f"Wave spectrum: Hs={args.hs} m, Tp={args.tp} s, gamma={args.gamma}")

        # Compute statistics
        stats = predict.statistics(tf, wave, duration=args.duration)

        # Print results
        print(f"\nResponse Statistics (duration={args.duration}s):")
        print("-" * 60)
        for var, var_stats in stats.items():
            print(f"\n{var}:")
            print(f"  Hs (4*sqrt(m0)): {var_stats['hs']:.4f}")
            print(f"  Tz:              {var_stats['tz']:.2f} s")
            print(f"  Tp:              {var_stats['tp']:.2f} s")
            print(f"  MPM:             {var_stats['mpm']:.4f}")
            print(f"  Sigma:           {var_stats['sigma']:.4f}")

        # Save to JSON if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\nSaved results to: {output_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Handle generate command."""
    from spectraflex.orcaflex import white_noise

    template = Path(args.template)

    if not template.exists():
        print(f"Error: Template file not found: {template}", file=sys.stderr)
        return 1

    freq_range = tuple(args.freq_range)
    output_dir = Path(args.output_dir)

    print(f"Generating white noise models from: {template}")
    print(f"  Hs: {args.hs}")
    print(f"  Directions: {args.direction}")
    print(f"  Frequency range: {freq_range[0]} - {freq_range[1]} Hz")
    print(f"  Duration: {args.duration} s")

    try:
        if len(args.hs) == 1 and len(args.direction) == 1:
            # Single case
            path = white_noise.generate(
                template=template,
                hs=args.hs[0],
                freq_range=freq_range,
                duration=args.duration,
                wave_direction=args.direction[0],
                output_dir=output_dir,
            )
            print(f"\nGenerated: {path}")
        else:
            # Batch
            paths = white_noise.generate_batch(
                template=template,
                matrix={
                    "hs": args.hs,
                    "wave_direction": args.direction,
                },
                freq_range=freq_range,
                duration=args.duration,
                output_dir=output_dir,
            )
            print(f"\nGenerated {len(paths)} files in: {output_dir}")
            for p in paths:
                print(f"  {p.name}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_library(args: argparse.Namespace) -> int:
    """Handle library command."""
    if args.library_command == "info":
        return cmd_library_info(args)
    elif args.library_command == "build":
        return cmd_library_build(args)
    else:
        print("Error: Specify a library subcommand (info, build)", file=sys.stderr)
        return 1


def cmd_library_info(args: argparse.Namespace) -> int:
    """Show library information."""
    from spectraflex import TransferFunctionLibrary

    lib_path = Path(args.library_file)

    if not lib_path.exists():
        print(f"Error: Library file not found: {lib_path}", file=sys.stderr)
        return 1

    try:
        lib = TransferFunctionLibrary.load(lib_path)

        print(f"Library: {lib_path}")
        print(f"  Configurations: {len(lib)}")
        print(f"  Config keys: {lib.config_keys}")

        if len(lib) > 0:
            # Show ranges
            print("\nParameter ranges:")
            for key in lib.config_keys:
                min_val, max_val = lib.get_config_range(key)
                unique = lib.get_unique_values(key)
                print(f"  {key}: {min_val} to {max_val} ({len(unique)} values)")

            # Show first dataset info
            ds = lib.datasets[0]
            print("\nTransfer function shape:")
            print(f"  Frequencies: {len(ds.coords['frequency'])}")
            print(f"  Variables: {list(ds.coords['variable'].values)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_library_build(args: argparse.Namespace) -> int:
    """Build library from spectra files."""
    from spectraflex import TransferFunctionLibrary, identify
    from spectraflex.orcaflex.batch import config_from_filename

    spectra_dir = Path(args.spectra_dir)
    output_path = Path(args.output)

    if not spectra_dir.exists():
        print(f"Error: Directory not found: {spectra_dir}", file=sys.stderr)
        return 1

    spectra_files = sorted(spectra_dir.glob("*_spectra.npz"))

    if not spectra_files:
        print(f"Error: No *_spectra.npz files found in: {spectra_dir}", file=sys.stderr)
        return 1

    print(f"Building library from {len(spectra_files)} spectra files")

    try:
        lib = TransferFunctionLibrary()

        for path in spectra_files:
            config = config_from_filename(path)
            tf = identify.from_spectra(path, config=config)
            lib.add(tf)
            print(f"  Added: {path.name} -> {config}")

        lib.save(output_path)
        print(f"\nSaved library to: {output_path}")
        print(f"  Total configurations: {len(lib)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
