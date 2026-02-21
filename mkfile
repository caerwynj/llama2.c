<../inferno64/mkconfig

TARG=llmfs

OFILES=\
llmfs.$O\

HFILES=\
../inferno64/include/styxserver.h\

LIBS=styx 9 math

BIN=$ROOT/$OBJDIR/bin

<../inferno64/mkfiles/mkfile-$SYSTARG-$OBJTYPE

<../inferno64/mkfiles/mkone-$SHELLTYPE

CFLAGS=$CFLAGS -I../inferno64/include -I../inferno64 -I.

