#! /bin/bash
if [ -z "${OPENMM_HIP_INSTALLED}" ]; then
    openmm_version=$(pip show openmm | awk '/^Version:/ {print $2}')
    hip_version_major=$(cat /opt/rocm/.info/version | cut -d. -f1)
    hip_version_minor=$(cat /opt/rocm/.info/version | cut -d. -f2)
    torch_version=$(pip show torch | awk '/^Version:/ {print $2}')
    echo "will install openmm $openmm_version and pytorch $torch_version with HIP $hip_version_major.$hip_version_minor" &> rocm_setup_output.log
    pip3 install --force-reinstall                                                                 \
        openmm[hip${hip_version_major}]==${openmm_version} &>> rocm_setup_output.log
    pip3 install --force-reinstall                                                                 \
        torch==${torch_version}                                                                    \
        --index-url https://download.pytorch.org/whl/rocm${hip_version_major}.${hip_version_minor} &>> rocm_setup_output.log
fi
