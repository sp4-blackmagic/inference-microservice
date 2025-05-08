{
  inputs = {
    mach-nix.url = "github:DavHau/mach-nix";
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
          fastapi
          httpx
          rich
          uvicorn
          python-multipart
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


