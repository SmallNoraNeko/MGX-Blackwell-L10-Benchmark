#!/bin/bash
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Enable P2P specific ACS bits on every device that supports it

PLATFORM=$(dmidecode --string system-product-name)
logger "PLATFORM=${PLATFORM}"

# must be root to access extended PCI config space
if [ "$EUID" -ne 0 ]; then
  echo "ERROR: $0 must be run as root"
  exit 1
fi

for BDF in `lspci -d "*:*:*" | awk '{print $1}'`; do

    # skip if it doesn't support ACS
    setpci -v -s ${BDF} ECAP_ACS+0x6.w > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        # echo "${BDF} does not support ACS, skipping"
        continue
    fi

    logger "Disabling ACS on `lspci -s ${BDF}`"
    setpci -v -s ${BDF} ECAP_ACS+0x6.w
    setpci -v -s ${BDF} ECAP_ACS+0x6.w=0x0
    setpci -v -s ${BDF} ECAP_ACS+0x6.w
    if [ $? -ne 0 ]; then
        logger "Error enabling ACS on ${BDF}"
        continue
    fi
    NEW_VAL=`setpci -v -s ${BDF} ECAP_ACS+0x6.w | awk '{print $NF}'`
    # Convert hex to decimal for numeric comparison (handles 0x0, 0000, 0x0000, etc.)
    # ${NEW_VAL#0x} removes the 0x prefix if present
    # 16# tells bash to interpret it as decimal
    NEW_VAL_DEC=$((16#${NEW_VAL#0x}))
    if [ "${NEW_VAL_DEC}" -ne 0 ]; then
        logger "Failed to Disable ACS on ${BDF}"
        continue
    fi
done

