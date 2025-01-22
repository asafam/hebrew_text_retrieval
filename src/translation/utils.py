class SafeDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"  # Keep unresolved placeholders