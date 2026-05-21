try:
    from RPLCD.i2c import CharLCD
    lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)
    lcd.clear()
    lcd_available = True
except Exception as e:
    print(f"LCD Initialization failed: {e}")
    lcd_available = False

LANG_NAMES = {
    'ar': 'Arabic', 'ca': 'Catalan', 'de': 'German', 'en': 'English', 
    'es': 'Spanish', 'fr': 'French', 'it': 'Italian', 'nl': 'Dutch', 
    'uk': 'Ukrainian', 'zh': 'Chinese'
}

def update_lcd_display(source_code, target_code):
    if not lcd_available: return
    try:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        source_name = LANG_NAMES.get(source_code, source_code)
        # Guest is source
        lcd.write_string(f"Guest: {source_name[:9]}") 
        lcd.cursor_pos = (1, 0)
        target_name = LANG_NAMES.get(target_code, target_code)
        # User is target
        lcd.write_string(f"User:  {target_name[:9]}")
    except Exception as e:
        print(f"LCD Update failed: {e}")

def update_lcd_status(line1, line2):
    if not lcd_available: return
    try:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string(str(line1)[:16])
        lcd.cursor_pos = (1, 0)
        lcd.write_string(str(line2)[:16])
    except Exception as e:
        print(f"LCD Status Update failed: {e}")
