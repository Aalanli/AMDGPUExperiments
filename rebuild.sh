if [ -d "build" ]; then
    rm -r "build"
fi

mkdir "build"
cd "build"
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DGPU_TARGETS="gfx90a" ..
cp compile_commands.json ..
make