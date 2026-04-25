

class UiConfig:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port if args.port else (7861 if getattr(args, 'review', False) else 7860)
        self.review = getattr(args, 'review', False)
        self.audio_folder = getattr(args, 'audio_folder', None)

    def __str__(self):
        return ", ".join(f"{key}={value}" for key, value in self.__dict__.items())