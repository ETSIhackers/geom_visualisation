# Note/idea on this contribution
Enhance the viewer
  x Better viewer
    ~ add direction of detectors
    x view fov
  ~ use case to check geo ok?
  ~ Extract lut and save it
  x displaying normalisation
    x absolute
    ~ Show relative effeciency, alpha
  - Show modules only
  - Color hierachie module/detecteur


TODO: remove what following?
# PETSIRD template for use cases

The purpose of this repo is to provide a starting point for developing software that uses PETSIRD.
It is currently mostly needed/useful for C++ development, but is also useful for Python developers
who want to experiment with development of PETSIRD itself.

## Background
The [Emission Tomography Standardization Initiative (ETSI)](https://etsinitiative.org/)
is working towards establishing a standard for PET Raw Data, called PETSIRD ("PET ETSI Raw Data").
More information is on https://github.com/ETSInitiative/PETSIRD.

## How to use this template?

These instructions will use `YourRepoName` for the name of your new repository. Obviously replace it with something appropriate.

#### Create a new repository based on this template

Easiest is to start from GitHub:
1. Navigate to the URL of this repo: https://github.com/ETSInitiative/PETSIRDUseCaseTemplate
2. Click on the `Use this template` button and create your own repo
3. Pull it to your own local machine and modify
   1. Search-and-replace all occurences of `PETSIRDUseCaseTemplate` with `YourRepoName`
   2. Update the README.md to remove references to this template and write something about what your repo is going to do
   3. Update the `environment.yml`to include what you need. For instance, if you need ROOT, add something like `- root=6.28.0`
   4. Make some other basic changes and commit
      ```sh
      git commit -a -m "Updated template to YourRepoName"
      git push
      ```

### Using your repo

1. Open ***your*** repo in [GitHub Codespaces](https://code.visualstudio.com/docs/remote/codespaces) or
in a [VS Code devcontainer](https://code.visualstudio.com/docs/devcontainers/containers).
This codespace/container will contain all necessary tools, including `yardl` itself, as well as your repository.
2. Use `yardl` to generate C++ and Python code for the model:
  ```sh
  cd YourRepoName
  cd PETSIRD/model
  yardl generate
  cd ../..
  ```
3. Start working in either the [`cpp`](cpp/README.md) and/or [`python`](python/README.md) directories.
