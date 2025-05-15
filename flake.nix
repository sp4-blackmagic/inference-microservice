{
  description = "A FastAPI microservice for running inteference";

  inputs = {
    mach-nix.url = "github:DavHau/mach-nix";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { self
    , nixpkgs
    , mach-nix
    }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      inherit (mach-nix.lib.${system}) mkPython;

      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        pkgs.python3Packages.pip
      ]);
      # python = pkgs.python3.buildEnv.override {
      #   extraLibs = with pkgs.python3Packages; [
      #     # fastapi
      #     # fastapi-cli
      #     # httpx
      #     # rich
      #     # uvicorn
      #     # python-multipart
      #     # dask
      #     # pytest
      #     # joblib
      #     # numpy
      #     # pandas
      #     # toml
      #     # scikit-learn
      #     # imblearn
      #   ];
      #   ignoreCollisions = true;
      # };

    in
    {
      devShell.${system} = pkgs.mkShell {
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
      };
    };
}


