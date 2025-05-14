{
  description = "A FastAPI microservice for running inteference";

  inputs = {
    mach-nix.url = "github:DavHau/mach-nix";
    # nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
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
      python = pkgs.python3.buildEnv.override {
        extraLibs = with pkgs.python3Packages; [
          fastapi-cli
          fastapi
          httpx
          rich
          uvicorn
          python-multipart
          dask
          pytest
          joblib
          numpy
          pandas
          toml
          scikit-learn
        ];
        ignoreCollisions = true;
      };

    in
    {
      devShell.${system} = pkgs.mkShell {
        buildInputs = [
          python
        ];
      };
    };
}


