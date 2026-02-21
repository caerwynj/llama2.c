.TH LLMFS 3 
.SH NAME
llmfs \- interface to LLM file system
.SH SYNOPSIS
.B mount host /n/llm
.PP
.B /n/llm/clone
.br
.BI /n/llm/ n /ctl
.br
.BI /n/llm/ n /data
.SH DESCRIPTION
.I Llmfs
provides a way to interact with an LLM over a filesystem interface.
It serves a three-level directory.
The top of the hierarchy is a directory
.BR /n/llm ,
that contains a
.B clone
file and zero or more numbered directories.
Each directory represents a distinct connection to the LLM.
The directory contains two files:
.BR ctl ,
.BR data .
used as described below.
Opening the
.B clone
file reserves a connection: it is equivalent to opening the
.B ctl
file of an unused connection directory, creating a new one if necessary.
.PP
The file
.B ctl
controls a connection.
When read, it returns the decimal number
.I n
of its connection directory.
Thus, opening and reading
.B clone
allocates a connection directory and reveals the number of the allocated directory,
allowing the other files to be named (eg,
.BI /n/llm/ n /data\fR).
.PP
.B Ctl
accepts the following textual commands, allowing quoting as interpreted by
.IR parsecmd (10.2):
.TP
.BI "temp " n
Set the temperature of the LLM to
.IR n ,
which is number between 0 and 1.
Issue this request before starting prompt.
.TP
.BI "top " n
Select from the top n% of words
.IR n ,
which is a number between 0 and 1.
If write returns successfully, the command has started, and its standard input and
output may be accessed through
.BR data .
.PP
The
.B data
file provides a connection to the input and output of a LLM.
It must be opened separately for reading and for writing.
When opened for reading, it returns data that the LLM writes to its standard output; when closed, further writes by the command will receive the host
equivalent of `write to closed pipe'.
When opened for writing, data written to the file
is treated as a prompt to the LLM; when closed, further reads by
the LLM will see the host equivalent of `end of file'.
(Unfortunately there is no way to know when the command needs input.)
.PP
Once started, a LLM runs until it terminates or until it is killed,
by closing all
.BR ctl ,
and
.B data
files for a connection.
.PP
The read-only
.B status
file provides a single line giving the status of the connection (not the command), of the form:
.IP
.BI cmd/ "n opens state wdir arg0"
.PP
where the fields are separated by white space. The meaning of each field is:
.TP
.I n
The
.B LLM
directory number.
.TP
.I opens
The decimal number of open file descriptors for
.BR ctl ,
and
.BR data .
.SH SOURCE
.B llmfs.c
.SH DIAGNOSTICS
A
.B write
to
.B ctl
returns with an error and sets the error string if
a command cannot be started or killed successfully.
