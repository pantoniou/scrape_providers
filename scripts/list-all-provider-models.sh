#!/bin/sh
for p in `scrape-providers --list-providers`; do
	echo "# ${p}"
	scrape-providers --list-provider-models "$p" | sort
done
