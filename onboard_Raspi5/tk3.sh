#!/bin/sh

# ------------------------
# TK3 Quadrotor startup script
# ------------------------


# Add pocolibs binaries and libraries to PATH and LD_LIBRARY_PATH
export ROBOTPKG_BASE=/home/rpi/dvl
export PATH=$PATH:$ROBOTPKG_BASE/bin
export GENOMIX_SYSDIR=$ROBOTPKG_BASE/lib/genomix
export LD_LIBRARY_PATH=$ROBOTPKG_BASE/lib/genom/pocolibs:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$ROBOTPKG_BASE/lib/pkgconfig:${PKG_CONFIG_PATH}
pyver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
export PYTHONPATH=$ROBOTPKG_BASE/lib/python$pyver/site-packages:${PYTHONPATH}


# Settings
middleware=pocolibs  # or ros
components="
  rotorcraft
  nhfc
"

# Full paths (adjust if your binaries are elsewhere)
GENOMIX_BIN=/home/rpi/dvl/robotpkg/robots/genomix/src/genomixd
ROTORCRAFT_BIN=~/dvl/bin/rotorcraft-pocolibs
NHFC_BIN=~/dvl/bin/nhfc-pocolibs

# List of process IDs for cleanup
pids=

echo "robotpkg environment loaded"
echo "ROBOTPKG_BASE: $ROBOTPKG_BASE"
echo "python version: $(python3 --version)"

# --- Clean up stale PID files before starting ---
rm -f ~/.rotorcraft.pid-* ~/.nhfc.pid-*

echo "Initializing H2 middleware..."
h2 init

echo "Initializing pocolibs devices..."
pocolibs init || true
echo "OK"

echo "Starting Genomix server..."
genomixd &
GENOMIXD_PID=$!
sleep 2

echo "Starting component rotorcraft..."
rotorcraft-pocolibs &
ROTORCRAFT_PID=$!

echo "Starting component nhfc..."
nhfc-pocolibs &
NHFC_PID=$!

# --- Shutdown handler ---
cleanup() {
    echo "Shutting down processes..."
    kill $ROTORCRAFT_PID $NHFC_PID $GENOMIXD_PID 2>/dev/null || true
    killall rotorcraft-pocolibs nhfc-pocolibs genomixd 2>/dev/null || true
    rm -f ~/.rotorcraft.pid-* ~/.nhfc.pid-*
    echo "Ending H2 middleware..."
    h2 end
    echo "Removing pocolibs devices..."
    pocolibs end || true
    echo "OK"
}
trap cleanup EXIT INT TERM

# Keep script running so processes don’t exit immediately
wait
