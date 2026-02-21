#!/bin/sh

o.out&
inferno run test.sh
pkill o.out
