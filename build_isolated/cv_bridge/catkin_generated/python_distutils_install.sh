#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/xhy/catkin_ws/src/cv_bridge"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/xhy/catkin_ws/install_isolated/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/xhy/catkin_ws/install_isolated/lib/python3/dist-packages:/home/xhy/catkin_ws/build_isolated/cv_bridge/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/xhy/catkin_ws/build_isolated/cv_bridge" \
    "/home/xhy/xhy_env/bin/python3.8" \
    "/home/xhy/catkin_ws/src/cv_bridge/setup.py" \
     \
    build --build-base "/home/xhy/catkin_ws/build_isolated/cv_bridge" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/xhy/catkin_ws/install_isolated" --install-scripts="/home/xhy/catkin_ws/install_isolated/bin"
