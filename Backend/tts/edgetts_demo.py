import asyncio
import random
import edge_tts
import re

# https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/voices/list?trustedclienttoken=6A5AA1D4EAFF4E9FB37E23D68491D6F4
SUPPORTED_VOICES = {
'af-ZA-AdriNeural':'Female-Friendly-Positive',
'af-ZA-WillemNeural':'Male-Friendly-Positive',
'sq-AL-AnilaNeural':'Female-Friendly-Positive',
'sq-AL-IlirNeural':'Male-Friendly-Positive',
'am-ET-AmehaNeural':'Male-Friendly-Positive',
'am-ET-MekdesNeural':'Female-Friendly-Positive',
'ar-DZ-AminaNeural':'Female-Friendly-Positive',
'ar-DZ-IsmaelNeural':'Male-Friendly-Positive',
'ar-BH-AliNeural':'Male-Friendly-Positive',
'ar-BH-LailaNeural':'Female-Friendly-Positive',
'ar-EG-SalmaNeural':'Female-Friendly-Positive',
'ar-EG-ShakirNeural':'Male-Friendly-Positive',
'ar-IQ-BasselNeural':'Male-Friendly-Positive',
'ar-IQ-RanaNeural':'Female-Friendly-Positive',
'ar-JO-SanaNeural':'Female-Friendly-Positive',
'ar-JO-TaimNeural':'Male-Friendly-Positive',
'ar-KW-FahedNeural':'Male-Friendly-Positive',
'ar-KW-NouraNeural':'Female-Friendly-Positive',
'ar-LB-LaylaNeural':'Female-Friendly-Positive',
'ar-LB-RamiNeural':'Male-Friendly-Positive',
'ar-LY-ImanNeural':'Female-Friendly-Positive',
'ar-LY-OmarNeural':'Male-Friendly-Positive',
'ar-MA-JamalNeural':'Male-Friendly-Positive',
'ar-MA-MounaNeural':'Female-Friendly-Positive',
'ar-OM-AbdullahNeural':'Male-Friendly-Positive',
'ar-OM-AyshaNeural':'Female-Friendly-Positive',
'ar-QA-AmalNeural':'Female-Friendly-Positive',
'ar-QA-MoazNeural':'Male-Friendly-Positive',
'ar-SA-HamedNeural':'Male-Friendly-Positive',
'ar-SA-ZariyahNeural':'Female-Friendly-Positive',
'ar-SY-AmanyNeural':'Female-Friendly-Positive',
'ar-SY-LaithNeural':'Male-Friendly-Positive',
'ar-TN-HediNeural':'Male-Friendly-Positive',
'ar-TN-ReemNeural':'Female-Friendly-Positive',
'ar-AE-FatimaNeural':'Female-Friendly-Positive',
'ar-AE-HamdanNeural':'Male-Friendly-Positive',
'ar-YE-MaryamNeural':'Female-Friendly-Positive',
'ar-YE-SalehNeural':'Male-Friendly-Positive',
'az-AZ-BabekNeural':'Male-Friendly-Positive',
'az-AZ-BanuNeural':'Female-Friendly-Positive',
'bn-BD-NabanitaNeural':'Female-Friendly-Positive',
'bn-BD-PradeepNeural':'Male-Friendly-Positive',
'bn-IN-BashkarNeural':'Male-Friendly-Positive',
'bn-IN-TanishaaNeural':'Female-Friendly-Positive',
'bs-BA-GoranNeural':'Male-Friendly-Positive',
'bs-BA-VesnaNeural':'Female-Friendly-Positive',
'bg-BG-BorislavNeural':'Male-Friendly-Positive',
'bg-BG-KalinaNeural':'Female-Friendly-Positive',
'my-MM-NilarNeural':'Female-Friendly-Positive',
'my-MM-ThihaNeural':'Male-Friendly-Positive',
'ca-ES-EnricNeural':'Male-Friendly-Positive',
'ca-ES-JoanaNeural':'Female-Friendly-Positive',
'zh-HK-HiuGaaiNeural':'Female-Friendly-Positive',
'zh-HK-HiuMaanNeural':'Female-Friendly-Positive',
'zh-HK-WanLungNeural':'Male-Friendly-Positive',
'zh-CN-XiaoxiaoNeural':'Female-Warm',
'zh-CN-XiaoyiNeural':'Female-Lively',
'zh-CN-YunjianNeural':'Male-Passion',
'zh-CN-YunxiNeural':'Male-Lively-Sunshine',
'zh-CN-YunxiaNeural':'Male-Cute',
'zh-CN-YunyangNeural':'Male-Professional-Reliable',
'zh-CN-liaoning-XiaobeiNeural':'Female-Humorous',
'zh-TW-HsiaoChenNeural':'Female-Friendly-Positive',
'zh-TW-YunJheNeural':'Male-Friendly-Positive',
'zh-TW-HsiaoYuNeural':'Female-Friendly-Positive',
'zh-CN-shaanxi-XiaoniNeural':'Female-Bright',
'hr-HR-GabrijelaNeural':'Female-Friendly-Positive',
'hr-HR-SreckoNeural':'Male-Friendly-Positive',
'cs-CZ-AntoninNeural':'Male-Friendly-Positive',
'cs-CZ-VlastaNeural':'Female-Friendly-Positive',
'da-DK-ChristelNeural':'Female-Friendly-Positive',
'da-DK-JeppeNeural':'Male-Friendly-Positive',
'nl-BE-ArnaudNeural':'Male-Friendly-Positive',
'nl-BE-DenaNeural':'Female-Friendly-Positive',
'nl-NL-ColetteNeural':'Female-Friendly-Positive',
'nl-NL-FennaNeural':'Female-Friendly-Positive',
'nl-NL-MaartenNeural':'Male-Friendly-Positive',
'en-AU-NatashaNeural':'Female-Friendly-Positive',
'en-AU-WilliamNeural':'Male-Friendly-Positive',
'en-CA-ClaraNeural':'Female-Friendly-Positive',
'en-CA-LiamNeural':'Male-Friendly-Positive',
'en-HK-SamNeural':'Male-Friendly-Positive',
'en-HK-YanNeural':'Female-Friendly-Positive',
'en-IN-NeerjaExpressiveNeural':'Female-Friendly-Positive',
'en-IN-NeerjaNeural':'Female-Friendly-Positive',
'en-IN-PrabhatNeural':'Male-Friendly-Positive',
'en-IE-ConnorNeural':'Male-Friendly-Positive',
'en-IE-EmilyNeural':'Female-Friendly-Positive',
'en-KE-AsiliaNeural':'Female-Friendly-Positive',
'en-KE-ChilembaNeural':'Male-Friendly-Positive',
'en-NZ-MitchellNeural':'Male-Friendly-Positive',
'en-NZ-MollyNeural':'Female-Friendly-Positive',
'en-NG-AbeoNeural':'Male-Friendly-Positive',
'en-NG-EzinneNeural':'Female-Friendly-Positive',
'en-PH-JamesNeural':'Male-Friendly-Positive',
'en-PH-RosaNeural':'Female-Friendly-Positive',
'en-SG-LunaNeural':'Female-Friendly-Positive',
'en-SG-WayneNeural':'Male-Friendly-Positive',
'en-ZA-LeahNeural':'Female-Friendly-Positive',
'en-ZA-LukeNeural':'Male-Friendly-Positive',
'en-TZ-ElimuNeural':'Male-Friendly-Positive',
'en-TZ-ImaniNeural':'Female-Friendly-Positive',
'en-GB-LibbyNeural':'Female-Friendly-Positive',
'en-GB-MaisieNeural':'Female-Friendly-Positive',
'en-GB-RyanNeural':'Male-Friendly-Positive',
'en-GB-SoniaNeural':'Female-Friendly-Positive',
'en-GB-ThomasNeural':'Male-Friendly-Positive',
'en-US-AvaMultilingualNeural':'Female-Expressive-Caring-Pleasant-Friendly',
'en-US-AndrewMultilingualNeural':'Male-Warm-Confident-Authentic-Honest',
'en-US-EmmaMultilingualNeural':'Female-Cheerful-Clear-Conversational',
'en-US-BrianMultilingualNeural':'Male-Approachable-Casual-Sincere',
'en-US-AvaNeural':'Female-Expressive-Caring-Pleasant-Friendly',
'en-US-AndrewNeural':'Male-Warm-Confident-Authentic-Honest',
'en-US-EmmaNeural':'Female-Cheerful-Clear-Conversational',
'en-US-BrianNeural':'Male-Approachable-Casual-Sincere',
'en-US-AnaNeural':'Female-Cute',
'en-US-AriaNeural':'Female-Positive-Confident',
'en-US-ChristopherNeural':'Male-Reliable-Authority',
'en-US-EricNeural':'Male-Rational',
'en-US-GuyNeural':'Male-Passion',
'en-US-JennyNeural':'Female-Friendly-Considerate-Comfort',
'en-US-MichelleNeural':'Female-Friendly-Pleasant',
'en-US-RogerNeural':'Male-Lively',
'en-US-SteffanNeural':'Male-Rational',
'et-EE-AnuNeural':'Female-Friendly-Positive',
'et-EE-KertNeural':'Male-Friendly-Positive',
'fil-PH-AngeloNeural':'Male-Friendly-Positive',
'fil-PH-BlessicaNeural':'Female-Friendly-Positive',
'fi-FI-HarriNeural':'Male-Friendly-Positive',
'fi-FI-NooraNeural':'Female-Friendly-Positive',
'fr-BE-CharlineNeural':'Female-Friendly-Positive',
'fr-BE-GerardNeural':'Male-Friendly-Positive',
'fr-CA-ThierryNeural':'Male-Friendly-Positive',
'fr-CA-AntoineNeural':'Male-Friendly-Positive',
'fr-CA-JeanNeural':'Male-Friendly-Positive',
'fr-CA-SylvieNeural':'Female-Friendly-Positive',
'fr-FR-VivienneMultilingualNeural':'Female-Friendly-Positive',
'fr-FR-RemyMultilingualNeural':'Male-Friendly-Positive',
'fr-FR-DeniseNeural':'Female-Friendly-Positive',
'fr-FR-EloiseNeural':'Female-Friendly-Positive',
'fr-FR-HenriNeural':'Male-Friendly-Positive',
'fr-CH-ArianeNeural':'Female-Friendly-Positive',
'fr-CH-FabriceNeural':'Male-Friendly-Positive',
'gl-ES-RoiNeural':'Male-Friendly-Positive',
'gl-ES-SabelaNeural':'Female-Friendly-Positive',
'ka-GE-EkaNeural':'Female-Friendly-Positive',
'ka-GE-GiorgiNeural':'Male-Friendly-Positive',
'de-AT-IngridNeural':'Female-Friendly-Positive',
'de-AT-JonasNeural':'Male-Friendly-Positive',
'de-DE-SeraphinaMultilingualNeural':'Female-Friendly-Positive',
'de-DE-FlorianMultilingualNeural':'Male-Friendly-Positive',
'de-DE-AmalaNeural':'Female-Friendly-Positive',
'de-DE-ConradNeural':'Male-Friendly-Positive',
'de-DE-KatjaNeural':'Female-Friendly-Positive',
'de-DE-KillianNeural':'Male-Friendly-Positive',
'de-CH-JanNeural':'Male-Friendly-Positive',
'de-CH-LeniNeural':'Female-Friendly-Positive',
'el-GR-AthinaNeural':'Female-Friendly-Positive',
'el-GR-NestorasNeural':'Male-Friendly-Positive',
'gu-IN-DhwaniNeural':'Female-Friendly-Positive',
'gu-IN-NiranjanNeural':'Male-Friendly-Positive',
'he-IL-AvriNeural':'Male-Friendly-Positive',
'he-IL-HilaNeural':'Female-Friendly-Positive',
'hi-IN-MadhurNeural':'Male-Friendly-Positive',
'hi-IN-SwaraNeural':'Female-Friendly-Positive',
'hu-HU-NoemiNeural':'Female-Friendly-Positive',
'hu-HU-TamasNeural':'Male-Friendly-Positive',
'is-IS-GudrunNeural':'Female-Friendly-Positive',
'is-IS-GunnarNeural':'Male-Friendly-Positive',
'id-ID-ArdiNeural':'Male-Friendly-Positive',
'id-ID-GadisNeural':'Female-Friendly-Positive',
'ga-IE-ColmNeural':'Male-Friendly-Positive',
'ga-IE-OrlaNeural':'Female-Friendly-Positive',
'it-IT-GiuseppeNeural':'Male-Friendly-Positive',
'it-IT-DiegoNeural':'Male-Friendly-Positive',
'it-IT-ElsaNeural':'Female-Friendly-Positive',
'it-IT-IsabellaNeural':'Female-Friendly-Positive',
'ja-JP-KeitaNeural':'Male-Friendly-Positive',
'ja-JP-NanamiNeural':'Female-Friendly-Positive',
'jv-ID-DimasNeural':'Male-Friendly-Positive',
'jv-ID-SitiNeural':'Female-Friendly-Positive',
'kn-IN-GaganNeural':'Male-Friendly-Positive',
'kn-IN-SapnaNeural':'Female-Friendly-Positive',
'kk-KZ-AigulNeural':'Female-Friendly-Positive',
'kk-KZ-DauletNeural':'Male-Friendly-Positive',
'km-KH-PisethNeural':'Male-Friendly-Positive',
'km-KH-SreymomNeural':'Female-Friendly-Positive',
'ko-KR-HyunsuNeural':'Male-Friendly-Positive',
'ko-KR-InJoonNeural':'Male-Friendly-Positive',
'ko-KR-SunHiNeural':'Female-Friendly-Positive',
'lo-LA-ChanthavongNeural':'Male-Friendly-Positive',
'lo-LA-KeomanyNeural':'Female-Friendly-Positive',
'lv-LV-EveritaNeural':'Female-Friendly-Positive',
'lv-LV-NilsNeural':'Male-Friendly-Positive',
'lt-LT-LeonasNeural':'Male-Friendly-Positive',
'lt-LT-OnaNeural':'Female-Friendly-Positive',
'mk-MK-AleksandarNeural':'Male-Friendly-Positive',
'mk-MK-MarijaNeural':'Female-Friendly-Positive',
'ms-MY-OsmanNeural':'Male-Friendly-Positive',
'ms-MY-YasminNeural':'Female-Friendly-Positive',
'ml-IN-MidhunNeural':'Male-Friendly-Positive',
'ml-IN-SobhanaNeural':'Female-Friendly-Positive',
'mt-MT-GraceNeural':'Female-Friendly-Positive',
'mt-MT-JosephNeural':'Male-Friendly-Positive',
'mr-IN-AarohiNeural':'Female-Friendly-Positive',
'mr-IN-ManoharNeural':'Male-Friendly-Positive',
'mn-MN-BataaNeural':'Male-Friendly-Positive',
'mn-MN-YesuiNeural':'Female-Friendly-Positive',
'ne-NP-HemkalaNeural':'Female-Friendly-Positive',
'ne-NP-SagarNeural':'Male-Friendly-Positive',
'nb-NO-FinnNeural':'Male-Friendly-Positive',
'nb-NO-PernilleNeural':'Female-Friendly-Positive',
'ps-AF-GulNawazNeural':'Male-Friendly-Positive',
'ps-AF-LatifaNeural':'Female-Friendly-Positive',
'fa-IR-DilaraNeural':'Female-Friendly-Positive',
'fa-IR-FaridNeural':'Male-Friendly-Positive',
'pl-PL-MarekNeural':'Male-Friendly-Positive',
'pl-PL-ZofiaNeural':'Female-Friendly-Positive',
'pt-BR-ThalitaNeural':'Female-Friendly-Positive',
'pt-BR-AntonioNeural':'Male-Friendly-Positive',
'pt-BR-FranciscaNeural':'Female-Friendly-Positive',
'pt-PT-DuarteNeural':'Male-Friendly-Positive',
'pt-PT-RaquelNeural':'Female-Friendly-Positive',
'ro-RO-AlinaNeural':'Female-Friendly-Positive',
'ro-RO-EmilNeural':'Male-Friendly-Positive',
'ru-RU-DmitryNeural':'Male-Friendly-Positive',
'ru-RU-SvetlanaNeural':'Female-Friendly-Positive',
'sr-RS-NicholasNeural':'Male-Friendly-Positive',
'sr-RS-SophieNeural':'Female-Friendly-Positive',
'si-LK-SameeraNeural':'Male-Friendly-Positive',
'si-LK-ThiliniNeural':'Female-Friendly-Positive',
'sk-SK-LukasNeural':'Male-Friendly-Positive',
'sk-SK-ViktoriaNeural':'Female-Friendly-Positive',
'sl-SI-PetraNeural':'Female-Friendly-Positive',
'sl-SI-RokNeural':'Male-Friendly-Positive',
'so-SO-MuuseNeural':'Male-Friendly-Positive',
'so-SO-UbaxNeural':'Female-Friendly-Positive',
'es-AR-ElenaNeural':'Female-Friendly-Positive',
'es-AR-TomasNeural':'Male-Friendly-Positive',
'es-BO-MarceloNeural':'Male-Friendly-Positive',
'es-BO-SofiaNeural':'Female-Friendly-Positive',
'es-CL-CatalinaNeural':'Female-Friendly-Positive',
'es-CL-LorenzoNeural':'Male-Friendly-Positive',
'es-ES-XimenaNeural':'Female-Friendly-Positive',
'es-CO-GonzaloNeural':'Male-Friendly-Positive',
'es-CO-SalomeNeural':'Female-Friendly-Positive',
'es-CR-JuanNeural':'Male-Friendly-Positive',
'es-CR-MariaNeural':'Female-Friendly-Positive',
'es-CU-BelkysNeural':'Female-Friendly-Positive',
'es-CU-ManuelNeural':'Male-Friendly-Positive',
'es-DO-EmilioNeural':'Male-Friendly-Positive',
'es-DO-RamonaNeural':'Female-Friendly-Positive',
'es-EC-AndreaNeural':'Female-Friendly-Positive',
'es-EC-LuisNeural':'Male-Friendly-Positive',
'es-SV-LorenaNeural':'Female-Friendly-Positive',
'es-SV-RodrigoNeural':'Male-Friendly-Positive',
'es-GQ-JavierNeural':'Male-Friendly-Positive',
'es-GQ-TeresaNeural':'Female-Friendly-Positive',
'es-GT-AndresNeural':'Male-Friendly-Positive',
'es-GT-MartaNeural':'Female-Friendly-Positive',
'es-HN-CarlosNeural':'Male-Friendly-Positive',
'es-HN-KarlaNeural':'Female-Friendly-Positive',
'es-MX-DaliaNeural':'Female-Friendly-Positive',
'es-MX-JorgeNeural':'Male-Friendly-Positive',
'es-NI-FedericoNeural':'Male-Friendly-Positive',
'es-NI-YolandaNeural':'Female-Friendly-Positive',
'es-PA-MargaritaNeural':'Female-Friendly-Positive',
'es-PA-RobertoNeural':'Male-Friendly-Positive',
'es-PY-MarioNeural':'Male-Friendly-Positive',
'es-PY-TaniaNeural':'Female-Friendly-Positive',
'es-PE-AlexNeural':'Male-Friendly-Positive',
'es-PE-CamilaNeural':'Female-Friendly-Positive',
'es-PR-KarinaNeural':'Female-Friendly-Positive',
'es-PR-VictorNeural':'Male-Friendly-Positive',
'es-ES-AlvaroNeural':'Male-Friendly-Positive',
'es-ES-ElviraNeural':'Female-Friendly-Positive',
'es-US-AlonsoNeural':'Male-Friendly-Positive',
'es-US-PalomaNeural':'Female-Friendly-Positive',
'es-UY-MateoNeural':'Male-Friendly-Positive',
'es-UY-ValentinaNeural':'Female-Friendly-Positive',
'es-VE-PaolaNeural':'Female-Friendly-Positive',
'es-VE-SebastianNeural':'Male-Friendly-Positive',
'su-ID-JajangNeural':'Male-Friendly-Positive',
'su-ID-TutiNeural':'Female-Friendly-Positive',
'sw-KE-RafikiNeural':'Male-Friendly-Positive',
'sw-KE-ZuriNeural':'Female-Friendly-Positive',
'sw-TZ-DaudiNeural':'Male-Friendly-Positive',
'sw-TZ-RehemaNeural':'Female-Friendly-Positive',
'sv-SE-MattiasNeural':'Male-Friendly-Positive',
'sv-SE-SofieNeural':'Female-Friendly-Positive',
'ta-IN-PallaviNeural':'Female-Friendly-Positive',
'ta-IN-ValluvarNeural':'Male-Friendly-Positive',
'ta-MY-KaniNeural':'Female-Friendly-Positive',
'ta-MY-SuryaNeural':'Male-Friendly-Positive',
'ta-SG-AnbuNeural':'Male-Friendly-Positive',
'ta-SG-VenbaNeural':'Female-Friendly-Positive',
'ta-LK-KumarNeural':'Male-Friendly-Positive',
'ta-LK-SaranyaNeural':'Female-Friendly-Positive',
'te-IN-MohanNeural':'Male-Friendly-Positive',
'te-IN-ShrutiNeural':'Female-Friendly-Positive',
'th-TH-NiwatNeural':'Male-Friendly-Positive',
'th-TH-PremwadeeNeural':'Female-Friendly-Positive',
'tr-TR-AhmetNeural':'Male-Friendly-Positive',
'tr-TR-EmelNeural':'Female-Friendly-Positive',
'uk-UA-OstapNeural':'Male-Friendly-Positive',
'uk-UA-PolinaNeural':'Female-Friendly-Positive',
'ur-IN-GulNeural':'Female-Friendly-Positive',
'ur-IN-SalmanNeural':'Male-Friendly-Positive',
'ur-PK-AsadNeural':'Male-Friendly-Positive',
'ur-PK-UzmaNeural':'Female-Friendly-Positive',
'uz-UZ-MadinaNeural':'Female-Friendly-Positive',
'uz-UZ-SardorNeural':'Male-Friendly-Positive',
'vi-VN-HoaiMyNeural':'Female-Friendly-Positive',
'vi-VN-NamMinhNeural':'Male-Friendly-Positive',
'cy-GB-AledNeural':'Male-Friendly-Positive',
'cy-GB-NiaNeural':'Female-Friendly-Positive',
'zu-ZA-ThandoNeural':'Female-Friendly-Positive',
'zu-ZA-ThembaNeural':'Male-Friendly-Positive'
}


SUPPORTED_LANGUAGES = {
    **dict(zip(SUPPORTED_VOICES.values(), SUPPORTED_VOICES.keys())),
}

CH_LANGUAGE_ID=[]
EN_LANGUAGE_ID=[]
for id,attr in SUPPORTED_VOICES.items():
    if 'zh-CN' in id:CH_LANGUAGE_ID.append(f'{id}:{attr}')
    if 'en-US' in id:EN_LANGUAGE_ID.append(f'{id}:{attr}')


async def tts(TEXT, VOICE,OUTPUT_FILE,rate="+0%",volume="+0%",pitch="+0Hz"):
    communicate = edge_tts.Communicate(TEXT, VOICE,
                                        rate=rate,
                                        volume=volume,
                                        pitch=pitch,)
    with open(OUTPUT_FILE, "wb") as file:
        async for chunk in communicate.stream():  # 流式获取
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                print(f"WordBoundary: {chunk}")

async def search_voice_tts(TEXT,OUTPUT_FILE):
    # 根据条件获取语音列表
    voices = await edge_tts.VoicesManager.create()
    # 查找男性、中文、中国大陆的语音
    voice = voices.find(Gender="Male", Language="zh", Locale="zh-CN")
    print(voice)
    # 在查找的结果中随机选择语音
    selected_voice = random.choice(voice)["Name"]
    print(selected_voice)
    communicate = edge_tts.Communicate(TEXT, random.choice(voice)["Name"])
    await communicate.save(OUTPUT_FILE)

async def tts_with_submaker(TEXT, VOICE,OUTPUT_FILE,WEBVTT_FILE,rate="+0%",volume="+0%",pitch="+0Hz"):
    """输出字幕"""
    communicate = edge_tts.Communicate(TEXT, VOICE,
                                        rate=rate,
                                        volume=volume,
                                        pitch=pitch,)
    submaker = edge_tts.SubMaker()
    with open(OUTPUT_FILE, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

    with open(WEBVTT_FILE, "w", encoding="utf-8") as file:
        file.write(submaker.generate_subs())

def sync_tts_with_submaker(TEXT, VOICE,OUTPUT_FILE,WEBVTT_FILE,rate="+0%",volume="+0%",pitch="+0Hz"):
    """Main function to process audio and metadata synchronously."""
    communicate = edge_tts.Communicate(TEXT, VOICE,
                                        rate=rate,
                                        volume=volume,
                                        pitch=pitch,)
    submaker = edge_tts.SubMaker()
    with open(OUTPUT_FILE, "wb") as file:
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])
    


    with open(WEBVTT_FILE, "w", encoding="utf-8") as file:
        #file.write(submaker.generate_subs())
        
        subs = re.sub(r'([\u4e00-\u9fa5]) +([\u4e00-\u9fa5])', r'\1\2', submaker.generate_subs())
        #print(subs)
        #file.write(subs)
        newtext=[]
        for text in subs.split('\n'):
            if '00' not in text:
                text = re.sub(r'([\u4e00-\u9fa5]) +([\u4e00-\u9fa5])', r'\1\2', text)
            newtext.append(text)
        newtext='\n'.join(newtext)
        #print(newtext)
        file.write(newtext)
            

class edgetts_api():
    def __init__(self):
        #self.voices_options = asyncio.run(edge_tts.list_voices())
        self.matureman_defaultvoice=['zh-CN-YunjianNeural',]
        self.youngman_defaultvoice=['zh-CN-YunxiNeural']
        self.boy_defaultvoice=['zh-CN-YunxiaNeural']
        self.woman_defaultvoice=['zh-CN-XiaoxiaoNeural',]
        self.girl_defaultvoice=['zh-CN-XiaoyiNeural']
        self.woman_shanxi_defaultvoice=['zh-CN-shaanxi-XiaoniNeural',]

    def base_tts(self,text,outputfile,rate='+0%',webvitfile=None,voice='zh-CN-YunxiNeural'):
        #loop = asyncio.get_event_loop_policy().get_event_loop()
        voice=voice.split(':')[0]
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError as e:
            if str(e).startswith('There is no current event loop in thread'):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                raise
        try:
            if webvitfile is None:
                loop.run_until_complete(tts(text,voice,outputfile,rate=rate))
            else:
                loop.run_until_complete(tts_with_submaker(text,voice,outputfile,webvitfile,rate=rate))
        finally:
            loop.close()
    
    def sync_tts(self,text,outputfile,webvitfile,voice='zh-CN-YunxiNeural',rate=0):
        voice=voice.split(':')[0]
        #communicate = edge_tts.Communicate(text, voice)
        #communicate.save_sync(outputfile)
        if rate>=0:
            rate=f'+{rate}%'
        else:
            rate=f'{rate}%'
        sync_tts_with_submaker(text,voice,outputfile,webvitfile,rate=rate)

if __name__ == "__main__":
    TEXT = "微软的 edge tts 好棒啊!"
    VOICE = "zh-CN-YunyangNeural"  # ShortName
    OUTPUT_FILE = "test1.mp3"
    WEBVTT_FILE = "test3.vtt"
    # 列出相关的voice
    voices_options = asyncio.run(edge_tts.list_voices())
    allvoice_dict={}
    #tmpfile=open('voices.txt','w',encoding='utf-8')
    for voice in voices_options:
        VOICE=voice['ShortName']
        gender=voice['Gender']
        personality='-'.join(voice['VoiceTag']['VoicePersonalities'])
        allvoice_dict[VOICE]=f'{gender}-{personality}'
        #tmpfile.write(f"'{VOICE}':'{gender}-{personality}',\n")
    print(allvoice_dict)
    
    #voices_options = [voice for voice in voices_options if voice["Locale"].startswith("zh-")]
    #print(voices_options)
    
    # 调用 tts
    #for voice in voices_options:
    #    VOICE=voice['ShortName']
    #    asyncio.run(tts(TEXT,VOICE ,f"edgetts_demo/test_{VOICE}.mp3"))
    
    ## 调用 search_voice_tts, 随机选择语音
    #asyncio.run(search_voice_tts(TEXT,"test2.mp3"))
    # 调用 tts_with_submaker, 生成字幕
    api= edgetts_api()
    TEXT='总结一下，FFmpeg是一款功能强大的多媒体处理工具，可以用于处理音视频文件。'
    api.sync_tts(TEXT, "test.mp3","test.vtt")
    TEXT='i love a litte cat'
    #api.sync_tts(TEXT, "test.mp3","test.vtt")
    #api.sync_tts(TEXT, "test-50.mp3","test-50.vtt",rate='-50%')
    #api.sync_tts(TEXT, "test50.mp3","test50.vtt",rate='+50%')