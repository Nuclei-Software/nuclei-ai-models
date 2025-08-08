#!/bin/bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Utility script that handles downloading, extracting, and patching third-party
# library dependencies for TensorFlow Lite for Microcontrollers.
# Called with four arguments:
# 1 - URL to download from.
# 2 - MD5 checksum to verify the package's integrity. Use md5sum to create one.
# 3 - Path to new folder to unpack the library into.
# 4 - Optional patching action name.


RLSROOT=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

# Patches the Nuclei SDK to work around build issues with bigger ilm/dlm.
patch_nuclei_sdk() {
  local nsdk_dir="${1}"
  local nsdk_evalsoc_mem=${nsdk_dir}/SoC/evalsoc/Board/nuclei_fpga_eval/Source/GCC/evalsoc.memory
  echo "Patching Nuclei SDK, change evalsoc ilm link script file, ilm/dlm size changed from 64K to 512K"
  sed -i "s/\b0x10000\b/0x80000/g" $nsdk_evalsoc_mem
  echo "Finished preparing Nuclei SDK files"
}

# Main function handling the download, verify, extract, and patch process.
download_and_extract() {
  local usage="Usage: download_and_extract URL MD5 DIR [ACTION] [ACTION_PARAM]"
  local url="${1:?${usage}}"
  local expected_md5="${2:?${usage}}"
  local dir="${3:?${usage}}"
  local action=${4}
  local action_param1=${5}  # optional action parameter
  local tempdir=$(mktemp -d)
  local tempdir2=$(mktemp -d)
  local tempfile=${tempdir}/temp_file
  local curl_retries=5

  # Destionation already downloaded.
  if [ -d ${dir} ]; then
      return 0
  fi

  command -v curl >/dev/null 2>&1 || {
    echo >&2 "The required 'curl' tool isn't installed. Try 'apt-get install curl'."; return 1;
  }

  echo "downloading ${url}" >&2
  mkdir -p "${dir}"
  # We've been seeing occasional 56 errors from valid URLs, so set up a retry
  # loop to attempt to recover from them.
  for (( i=1; i<=$curl_retries; ++i )); do
    # We have to use this approach because we normally halt the script when
    # there's an error, and instead we want to catch errors so we can retry.

    curl -LsS --fail --retry 5 "${url}" > ${tempfile}
    CURL_RESULT=$?

    # Was the command successful? If so, continue.
    if [[ $CURL_RESULT -eq 0 ]]; then
      break
    fi

    # Keep trying if we see the '56' error code.
    if [[ ( $CURL_RESULT -ne 56 ) || ( $i -eq $curl_retries ) ]]; then
      echo "Error $CURL_RESULT downloading '${url}'"
      rm -rf "${dir}"
      return 1
    fi
    sleep 2
  done

  # Check that the file was downloaded correctly using a checksum. Put expected_md5 as "SKIP_MD5_CHECK" to skip this check.
  DOWNLOADED_MD5=$(openssl dgst -md5 ${tempfile} | sed 's/.* //g')
  if [ ${expected_md5} != ${DOWNLOADED_MD5} ] && [ ${expected_md5} != "SKIP_MD5_CHECK" ]; then
    echo "Checksum error for '${url}'. Expected ${expected_md5} but found ${DOWNLOADED_MD5}"
    rm -rf "${dir}"
    return 1
  fi

  # delete anything after the '?' in a url that may mask true file extension
  url=$(echo "${url}" | sed "s/\?.*//")

  if [[ "${url}" == *gz ]]; then
    tar -C "${tempdir2}" -xzf ${tempfile}
  elif [[ "${url}" == *tar.xz ]]; then
    tar -C "${tempdir2}" -xf ${tempfile}
  elif [[ "${url}" == *bz2 ]]; then
    curl -Ls "${url}" > ${tempdir}/tarred.bz2
    tar -C "${tempdir2}" -xjf ${tempfile}
  elif [[ "${url}" == *zip ]]; then
    unzip ${tempfile} -d ${tempdir2} 2>&1 1>/dev/null
  else
    echo "Error unsupported archive type. Failed to extract tool after download."
    rm -rf "${dir}"
    return 1
  fi

  # If the zip file contains nested directories, extract the files from the
  # inner directory.
  if [ $(find $tempdir2/* -maxdepth 0 | wc -l) = 1 ] && [ -d $tempdir2/* ]; then
    # Unzip to a temp dir, and move the files we want from the tempdir to destination.
    # We want this to be dependent on the folder structure of the zipped file, so --strip-components cannot be used.
    cp -R ${tempdir2}/*/* ${dir}/
  else
    cp -R ${tempdir2}/* ${dir}/
  fi

  rm -rf ${tempdir} ${tempdir2}

  # Delete any potential BUILD files, which would interfere with Bazel builds.
  find "${dir}" -type f -name '*BUILD' -delete

  if [[ ${action} == "patch_nuclei_sdk" ]]; then
    patch_nuclei_sdk ${dir}
  elif [[ ${action} ]]; then
    echo "Unknown action '${action}'"
    return 1
  fi
}

download_third_party() {
  NUCLEI_SDK_URL="https://github.com/Nuclei-Software/nuclei-sdk/archive/refs/tags/0.8.0.zip"
  NUCLEI_SDK_MD5="be0f1a52f1b7e5fa1a9d7c46e7d2fdb8"

  NUCLEI_STUDIO_URL="https://download.nucleisys.com/upload/files/nucleistudio/NucleiStudio_IDE_202502-lin64.tgz"
  NUCLEI_STUDIO_MD5="7efce78a7ae640e996ff2d2c7beed4c7"

  NUCLEI_NPK_TFLM="https://github.com/Nuclei-Software/npk-tflm/archive/refs/tags/0.5.0.zip"
  NUCLEI_NPK_TFLM_MD5="6ad9178533203accdce7df09082c01c2"

  if [ ! -d ${dir} ]; then
      mkdir -p downloads
  fi

  download_and_extract ${NUCLEI_STUDIO_URL} ${NUCLEI_STUDIO_MD5} downloads/nuclei_studio
  download_and_extract ${NUCLEI_SDK_URL} ${NUCLEI_SDK_MD5} downloads/nuclei-sdk patch_nuclei_sdk
  download_and_extract ${NUCLEI_NPK_TFLM} ${NUCLEI_NPK_TFLM_MD5} downloads/nuclei-sdk/Components/tflm
}

download_check() {
  if [ -d "downloads/nuclei_studio" ]; then
      echo "nuclei_studio already exists"
  fi
  if [ -d "downloads/nuclei-sdk" ]; then
      echo "nuclei-sdk already exists"
  fi
  if [ -d "downloads/nuclei-sdk/Components/tflm" ]; then
      echo "nuclei-sdk/Components/tflm already exists"
  fi
}

var_dupclean () {
    if [ "${1}" == "" ] ; then
        return
    fi
    local VARS=${1}
    local DELIM=${2:-:}
    VARS=$(echo $VARS | awk -v RS=$DELIM -v ORS=$DELIM '!a[$1]++ { if (NR > 1) printf ORS; printf $1 }')
    local NEWVARS
    for VAR in ${VARS//${DELIM}/ } ; do
        if ! [[ "${VAR// /}" == "" ]] ; then
            if [ "$NEWVARS" == "" ] ; then
                NEWVARS="${VAR}"
            else
                NEWVARS="${NEWVARS}${DELIM}${VAR}"
            fi
        fi
    done
    echo $NEWVARS
}

download_third_party

download_check

GCC_PATH=$RLSROOT/downloads/nuclei_studio/NucleiStudio/toolchain/gcc/bin
QEMU_PATH=$RLSROOT/downloads/nuclei_studio/NucleiStudio/toolchain/qemu/bin
OPENOCD_PATH=$RLSROOT/downloads/nuclei_studio/NucleiStudio/toolchain/openocd/bin

NEWPATH=${GCC_PATH}:${QEMU_PATH}:${OPENOCD_PATH}:${PATH}
export PATH=$(var_dupclean $NEWPATH)

echo -e "PATH is:" ${PATH}


