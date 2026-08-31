{
  description = "MoGe-2 WebGPU Spatial Scene development shell";

  # The channel tarball avoids GitHub API rate limits while the lock file still
  # pins the exact nixpkgs revision used by the development shell.
  inputs.nixpkgs.url = "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in {
      devShells = forAllSystems (system:
        let pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = [ pkgs.nodejs_24 ];
          };
        });
    };
}
