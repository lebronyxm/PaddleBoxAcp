# compile paddle code with gcc 8.2
cd build
function make_paddle() {
    make clean
    # if /tmp space is not enough, reduce num of make thread
    for i in `seq 0 5`
    do
        make -j 40 -s
    done
    [ $? -ne 0 ] && echo "paddle make failed! please run command [cd base_path/PaddleBox/build && make clean && make -j 40 >log.txt 2>&1 &] to see detail in file log.txt."
    echo "build ok!"
}

source /home/users/yangxuemeng/gcc8.bashrc

make_paddle
exit 0
