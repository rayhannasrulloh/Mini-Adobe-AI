import os
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from werkzeug.utils import secure_filename