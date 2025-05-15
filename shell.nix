{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
    pkgs.python3Packages.pip
  ]);
in
pkgs.mkShell {
  buildInputs = [ pythonEnv ];

  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.libz
  ];

  shellHook = ''
    export PYTHONPATH=$PWD
    if [ ! -d ".venv" ]; then
      python -m venv .venv
      source .venv/bin/activate
      pip install -r requirements.txt
    else
      source .venv/bin/activate
    fi
  '';
}

