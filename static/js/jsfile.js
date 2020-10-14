$(document).ready(function () {
    $('#table').hide();
    $('#confidence').hide();
    $('#process').hide();
    
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);


        // Show loading animation
        $(this).hide();
        $('.loader').show();
        // Make prediction by calling api /predict
        console.log(document.getElementById('choose').value);
        if(document.getElementById('choose').value == 'count') {
            $('#table').hide();
            $('#process').show();
            $('#confidence').hide();
               $.ajax({
                type: 'POST',
                url: '/predict',
                data: form_data,
                contentType: false,
                cache: false,
                processData: false,
                async: false,
                success: function (data) {
                    // Get and display the result
                    $('.loader').hide();
                    $('#count').fadeIn(600);
                    $('#count').text(' Total WBCs are:  ' + data);
                    console.log(data);

                    // var img = document.createElement('img');
                    // img.src = "../static/edgeM.png";
                    // document.getElementById('image').appendChild(img);

                },
            });
           }
           else if(document.getElementById('choose').value == 'classify') {
            $('#table').show();
            $('#confidence').show();
            $('#process').hide();
            $('#count').hide();
            $.ajax({
                type: 'POST',
                url: '/predict',
                data: form_data,
                contentType: false,
                cache: false,
                processData: false,
                async: false,
                success: function (data) {
                    // Get and display the result
                    $('.loader').hide();
                    $('#result').fadeIn(600);
                    $('#result').text(' Therefore, the required result is  ' + data.result);
                    console.log(data.probability[0]);
                    $('#eosino').text(data.probability[0]);
                    $('#lympho').text(data.probability[1]);
                    $('#mono').text(data.probability[2]);
                    $('#neutro').text(data.probability[3]);
                },
            });
        }
        // }
        // elseif(document.getElementById('choose').value === count) {
         
        // }
    });

});
