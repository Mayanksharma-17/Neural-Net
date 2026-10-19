{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    gcc
    python3 
    python313Packages.numpy
    python313Packages.pillow
    eigen
  ];
}
