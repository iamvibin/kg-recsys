#!/bin/bash

# Options can also be passed on the command line.
# These options are blind-passed to the CLI.
# Ex: ./run.sh -D log4j.threshold=DEBUG

readonly PSL_VERSION='2.2.2'
readonly JAR_PATH="./psl-cli-${PSL_VERSION}.jar"
readonly BASE_NAME='movie'
readonly baseline_id='000'
readonly splitId=$2
neighbour=$1
neigh=$1
printf -v neighbour "%03d" $neighbour
readonly ADDITIONAL_PSL_OPTIONS='--int-ids --postgres -D log4j.threshold=TRACE'
readonly ADDITIONAL_LEARN_OPTIONS='--learn'
readonly ADDITIONAL_EVAL_OPTIONS='--infer --eval ContinuousEvaluator'
readonly AVAILABLE_MEM_KB=$(cat /proc/meminfo | grep 'MemTotal' | sed 's/^[^0-9]\+\([0-9]\+\)[^0-9]\+$/\1/')
# Floor by multiples of 5 and then reserve an additional 5 GB.
readonly JAVA_MEM_GB=$((${AVAILABLE_MEM_KB} / 1024 / 1024 / 5 * 5 - 5))

function main() {
   trap exit SIGINT

   # The data is already included, no need to fetch it.

   # Make sure we can run PSL.
   check_requirements
   fetch_psl
   echo stat
   echo $1
   echo $splitId
   sed -i "s/baselineSplit/${baseline_id}_${splitId}/g" ${BASE_NAME}.data
   sed -i "s/neighbourSplit/${neighbour}_${splitId}/g" ${BASE_NAME}.data
   sed -i "s/NEIGHBOUR/${neigh}/g" ${BASE_NAME}.data

   # Run PSL
   #runWeightLearning "$@"
   #runEvaluation "$@"
   runEvaluationWithoutWL "$@"
   sed -i "s/${baseline_id}_${splitId}/baselineSplit/g" ${BASE_NAME}.data
   sed -i "s/${neighbour}_${splitId}/neighbourSplit/g" ${BASE_NAME}.data
   sed -i "s/${neigh}/NEIGHBOUR/g" ${BASE_NAME}.data
}
function runWeightLearning() {
   echo "Running PSL Weight Learning"

   java -jar "${JAR_PATH}" --model "${BASE_NAME}.psl" --data "${BASE_NAME}-learn.data" ${ADDITIONAL_LEARN_OPTIONS} ${ADDITIONAL_PSL_OPTIONS} "$@"
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: Failed to run weight learning'
      exit 60
   fi
}

function runEvaluation() {
   echo "Running PSL Inference"

   java -jar "${JAR_PATH}" --model "${BASE_NAME}-learned.psl" --data "${BASE_NAME}-eval.data" --output inferred-predicates ${ADDITIONAL_EVAL_OPTIONS} ${ADDITIONAL_PSL_OPTIONS} "$@"
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: Failed to run infernce'
      exit 70
   fi
}

function runEvaluationWithoutWL() {
   echo "Running PSL Inference"

   java -Xmx${JAVA_MEM_GB}G -Xms${JAVA_MEM_GB}G -jar "${JAR_PATH}" --model "${BASE_NAME}.psl" --data "${BASE_NAME}.data" --output inferred-predicates/${neighbour}_$splitId ${ADDITIONAL_EVAL_OPTIONS} ${ADDITIONAL_PSL_OPTIONS} "$@"
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: Failed to run infernce'
      exit 70
   fi
}

function check_requirements() {
   local hasWget
   local hasCurl

   type wget > /dev/null 2> /dev/null
   hasWget=$?

   type curl > /dev/null 2> /dev/null
   hasCurl=$?

   if [[ "${hasWget}" -ne 0 ]] && [[ "${hasCurl}" -ne 0 ]]; then
      echo 'ERROR: wget or curl required to download the jar'
      exit 10
   fi

   type java > /dev/null 2> /dev/null
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: java required to run project'
      exit 13
   fi
}

function get_fetch_command() {
   type curl > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "curl -o"
      return
   fi

   type wget > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "wget -O"
      return
   fi

   echo 'ERROR: wget or curl not found'
   exit 20
}

function fetch_file() {
   local url=$1
   local path=$2
   local name=$3

   if [[ -e "${path}" ]]; then
      echo "${name} file found cached, skipping download."
      return
   fi

   echo "Downloading ${name} file located at: '${url}'."
   `get_fetch_command` "${path}" "${url}"
   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to download ${name} file"
      exit 30
   fi
}

# Fetch the jar from a remote or local location and put it in this directory.
# Snapshots are fetched from the local maven repo and other builds are fetched remotely.
function fetch_psl() {
   if [[ $PSL_VERSION == *'SNAPSHOT'* ]]; then
      local snapshotJARPath="$HOME/.m2/repository/org/linqs/psl-cli/${PSL_VERSION}/psl-cli-${PSL_VERSION}.jar"
      cp "${snapshotJARPath}" "${JAR_PATH}"
   else
      local remoteJARURL="https://repo1.maven.org/maven2/org/linqs/psl-cli/${PSL_VERSION}/psl-cli-${PSL_VERSION}.jar"
      fetch_file "${remoteJARURL}" "${JAR_PATH}" 'psl-jar'
   fi
}

main "$@"
