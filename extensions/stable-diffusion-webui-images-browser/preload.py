

def preload(parser):
    parser.add_argument(
        "--image-browser-tmp-db",
        action="store_true",
        help="Copy database file to and from /tmp when transacting (workaround for filesystems sqlite does not support)"
    )